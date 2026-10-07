import json
import os
import re
import shutil
import sys
from typing import Any, Iterable, Optional
import time
import ray
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.config import PPOConfig
from verl.trainer.metrics import reduce_metrics
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.dataset import RLHFDataset, collate_fn


def _normalize_label_name(name: str) -> str:
    # return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    return name.replace("_", " ").replace("-", " ").replace(".", " ").lower()


def _extract_answer_text(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(text).strip()


def _compute_binary_label_accuracy(pred_text: str, ground_truth: str) -> float:
    pred = _normalize_label_name(_extract_answer_text(pred_text).strip())
    gt = _normalize_label_name(str(ground_truth).strip())
    return 1.0 if pred == gt else 0.0


def _snapshot_training_config(output_dir: str, config: Optional[PPOConfig] = None) -> None:
    """Save training script/command-line and config to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save command line arguments
    cmd_info = {
        "script": sys.argv[0],
        "command_line": " ".join(sys.argv),
        "arguments": sys.argv[1:],
    }
    with open(os.path.join(output_dir, "train_command.json"), "w", encoding="utf-8") as f:
        json.dump(cmd_info, f, indent=2, ensure_ascii=False)
    
    # Save config if provided
    if config is not None:
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            OmegaConf.save(config, f)


def _try_read_cmdline_and_cwd(pid: int) -> tuple[list[str], Optional[str]]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
        cmd = [x for x in raw.split("\x00") if x]
    except Exception:
        cmd = []
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        cwd = None
    return cmd, cwd


def _detect_launch_script_from_parents(max_depth: int = 4) -> Optional[str]:
    pid = os.getppid()
    for _ in range(max_depth):
        if pid <= 1:
            break
        cmd, cwd = _try_read_cmdline_and_cwd(pid)
        for token in reversed(cmd):
            if not token.endswith(".sh"):
                continue
            candidate = token
            if not os.path.isabs(candidate) and cwd is not None:
                candidate = os.path.join(cwd, candidate)
            candidate = os.path.abspath(candidate)
            if os.path.isfile(candidate):
                return candidate
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                stat_fields = f.read().split()
            pid = int(stat_fields[3])
        except Exception:
            break
    return None


def _snapshot_launch_script(cil_info_dir: str, explicit_script_path: Optional[str] = None) -> Optional[str]:
    os.makedirs(cil_info_dir, exist_ok=True)
    source_script = None

    if explicit_script_path:
        maybe = os.path.abspath(explicit_script_path)
        if os.path.isfile(maybe):
            source_script = maybe
        else:
            print(f"[CIL] Warning: --launch_script not found: {explicit_script_path}")

    if source_script is None:
        source_script = _detect_launch_script_from_parents()

    if source_script is None:
        print("[CIL] Warning: launch .sh script not detected; skip script snapshot.")
        return None

    target_path = os.path.join(cil_info_dir, 'train_script.sh')
    shutil.copy2(source_script, target_path)
    print(f"[CIL] Saved launch script to: {target_path}")
    return target_path


def _resolve_local_file(path_like: Optional[str]) -> Optional[str]:
    if not path_like:
        return None
    path_like = os.path.expanduser(str(path_like).strip())
    if not os.path.isabs(path_like):
        path_like = os.path.abspath(path_like)
    return path_like if os.path.isfile(path_like) else None


def _snapshot_repro_files(cil_info_dir: str, config: PPOConfig) -> None:
    os.makedirs(cil_info_dir, exist_ok=True)

    prompt_path = _resolve_local_file(getattr(config.data, "format_prompt", None))
    if prompt_path is not None:
        prompt_target = os.path.join(cil_info_dir, os.path.basename(prompt_path))
        shutil.copy2(prompt_path, prompt_target)
        print(f"[CIL] Saved prompt file to: {prompt_target}")
    else:
        print(f"[CIL] Warning: prompt file not found: {getattr(config.data, 'format_prompt', None)}")

    reward_spec = str(getattr(config.worker.reward, "reward_function", "") or "")
    reward_file = reward_spec.split(":", 1)[0].strip() if reward_spec else ""
    reward_path = _resolve_local_file(reward_file)
    if reward_path is not None:
        reward_target = os.path.join(cil_info_dir, os.path.basename(reward_path))
        shutil.copy2(reward_path, reward_target)
        print(f"[CIL] Saved reward file to: {reward_target}")
    else:
        print(f"[CIL] Warning: reward file not found: {reward_spec}")


def _count_class_samples(data_path: str, allowed_classes_norm: set[str]) -> int:
    if not os.path.isdir(data_path):
        raise ValueError(f"data_path '{data_path}' is not a directory; only folder-per-class datasets are supported in CIL mode.")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    total = 0
    for class_name in os.listdir(data_path):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        norm = _normalize_label_name(class_name)
        if norm not in allowed_classes_norm:
            continue
        for fname in os.listdir(class_dir):
            if os.path.splitext(fname)[1].lower() in exts:
                total += 1
    return total


def _build_dataloader(
    config_data,
    tokenizer,
    processor,
    allowed_classes: Optional[Iterable[str]],
    is_train: bool,
    label_list_override: Optional[list[str]] = None,
    allowed_class_order: Optional[list[str]] = None,
):
    # When allowed_classes is provided (CIL task), skip expensive global overlong-filtering on the full dataset;
    # we will filter after subsetting classes.
    filter_overlong = config_data.filter_overlong_prompts if allowed_classes is None else False

    dataset = RLHFDataset(
        data_path=config_data.train_files if is_train else config_data.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config_data.prompt_key,
        answer_key=config_data.answer_key,
        image_key=config_data.image_key,
        video_key=config_data.video_key,
        image_dir=config_data.image_dir,
        video_fps=config_data.video_fps,
        max_prompt_length=config_data.max_prompt_length,
        truncation="right",
        format_prompt=config_data.format_prompt,
        min_pixels=config_data.min_pixels,
        max_pixels=config_data.max_pixels,
        filter_overlong_prompts=filter_overlong,
        filter_overlong_prompts_workers=config_data.filter_overlong_prompts_workers,
        label_list=[_normalize_label_name(x) for x in label_list_override]
        if label_list_override is not None
        else None,
    )

    if allowed_classes is not None:
        allowed_classes_list = list(allowed_classes)
        allowed_classes_norm = {_normalize_label_name(c) for c in allowed_classes_list}
        dataset.dataset = dataset.dataset.filter(
            lambda example: _normalize_label_name(example[config_data.answer_key]) in allowed_classes_norm
        )
        labels_for_prompt = (
            [_normalize_label_name(x) for x in label_list_override]
            if label_list_override is not None
            else [_normalize_label_name(x) for x in (allowed_class_order or allowed_classes_list)]
        )
        dataset.label_list = labels_for_prompt

    if is_train:
        if config_data.shuffle:
            gen = torch.Generator()
            gen.manual_seed(config_data.seed)
            sampler = RandomSampler(data_source=dataset, generator=gen)
        else:
            sampler = SequentialSampler(data_source=dataset)
        batch_size = config_data.mini_rollout_batch_size or config_data.rollout_batch_size
        drop_last = True
    else:
        sampler = SequentialSampler(data_source=dataset)
        batch_size = len(dataset) if config_data.val_batch_size == -1 else config_data.val_batch_size
        drop_last = False

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=10,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=drop_last,
    )

    assert len(dataloader) >= 1
    return dataloader


class RayPPOContinualTrainer(RayPPOTrainer):
    def _load_checkpoint(self) -> None:
        # keep model loading, skip dataloader state to avoid sample-replay
        if self.config.trainer.load_checkpoint_path is None:
            return
        load_checkpoint_path = self.config.trainer.load_checkpoint_path
        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        reward_metrics_lst = {}
        length_metrics_lst = {}
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        collect_sample_level = bool(getattr(self, "collect_sample_level", False))
        sample_accuracy_scores: list[float] = []
        sample_labels_norm: list[str] = []
        cls_accuracy_scores: list[float] = []

        
        total_batches = len(self.val_dataloader) if hasattr(self.val_dataloader, "__len__") else None
        
        total_samples = len(getattr(self.val_dataloader, "dataset", [])) or "?"
        
        print(f"Start validation... ({total_samples} samples, {total_batches or '?'} batches)")
        sys.stdout.flush()

        val_t0 = time.monotonic()
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        val_batch_count = 0
        for val_batch_idx, batch_dict in enumerate(self.val_dataloader):
            val_batch_count = val_batch_idx + 1
            batch_t0 = time.monotonic()

            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            if val_batch_idx % 50 == 0 or (total_batches and val_batch_idx == total_batches - 1):
                elapsed = time.monotonic() - val_t0
                batch_time = time.monotonic() - batch_t0
                progress = f"{val_batch_idx + 1}/{total_batches}" if total_batches else f"{val_batch_idx + 1}"
                print(f"[VAL] batch {progress} | elapsed {elapsed:.0f}s | this batch {batch_time:.1f}s")
                sys.stdout.flush()

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))
            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst.setdefault(key, []).extend(value)

            batch_labels = test_batch.non_tensor_batch.get("ground_truth")
            output_ids = test_batch.batch.get("responses")
            if output_ids is not None and batch_labels is not None:
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                labels_text = [str(x) for x in batch_labels]
                if len(output_texts) == len(labels_text):
                    acc_list = [_compute_binary_label_accuracy(pred, gt) for pred, gt in zip(output_texts, labels_text)]
                    cls_accuracy_scores.extend(acc_list)
                    if collect_sample_level:
                        sample_accuracy_scores.extend(acc_list)
                        sample_labels_norm.extend([_normalize_label_name(x) for x in labels_text])

            for key, value in self._compute_length_metrics(test_batch).items():
                length_metrics_lst.setdefault(key, []).append(value)

            # Collect samples for optional generation logging (consistent with base trainer behavior)
            try:
                input_ids = test_batch.batch.get("prompts")
                output_ids = test_batch.batch.get("responses")
                labels = test_batch.non_tensor_batch.get("ground_truth")
                if input_ids is not None and output_ids is not None and labels is not None:
                    input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    scores = reward_tensor.sum(-1).cpu().tolist()
                    sample_inputs.extend(input_texts)
                    sample_outputs.extend(output_texts)
                    sample_labels.extend(labels)
                    sample_scores.extend(scores)
            except Exception:
                pass  # best effort; do not break validation

        self.actor_rollout_ref_wg.release_rollout_engine()
        
        if reward_tensor_lst:
            stacked = torch.cat(reward_tensor_lst, dim=0)
            self.val_reward_score = stacked.sum(-1).mean().item()
        else:
            self.val_reward_score = 0.0
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        val_elapsed = time.monotonic() - val_t0
        print(f"Finish validation. (total {val_elapsed:.0f}s, {val_batch_count} batches)")
        sys.stdout.flush()
        result = {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}
        result["val/cls_accuracy"] = float(np.mean(cls_accuracy_scores)) if cls_accuracy_scores else 0.0
        if collect_sample_level:
            result["_sample_accuracy_scores"] = sample_accuracy_scores
            result["_sample_labels_norm"] = sample_labels_norm
        return result

    def _compute_length_metrics(self, test_batch: DataProto) -> dict[str, Any]:
        # Avoid importing trainer.metrics to keep this file self contained
        lengths = {}
        input_lengths = [len(ids) for ids in test_batch.batch["input_ids"]]
        response_lengths = [len(ids) for ids in test_batch.batch["responses"]]
        lengths["input_len"] = sum(input_lengths) / len(input_lengths)
        lengths["response_len"] = sum(response_lengths) / len(response_lengths)
        return lengths

def _list_class_names_from_dir(data_path: str) -> list[str]:
    if not os.path.isdir(data_path):
        raise ValueError(
            f"data_path '{data_path}' is not a directory; only folder-per-class datasets are supported in CIL mode."
        )
    class_names = []
    for class_name in sorted(os.listdir(data_path)):
        if not os.path.isdir(os.path.join(data_path, class_name)):
            continue
        class_names.append(_normalize_label_name(class_name))
    if not class_names:
        raise ValueError(f"No class folders found under '{data_path}'.")
    return class_names


def _load_class_order_from_file(path: str) -> list[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Class order file {path} not found.")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".txt":
        with open(path, encoding="utf-8") as f:
            values = [line.strip() for line in f if line.strip()]
        try:
            return [int(v) for v in values]
        except ValueError:
            raise ValueError("Error: Invalid class order file format. Txt: each line a class ID; Json: 1D list.")
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or any(not isinstance(v, int) for v in data):
            raise ValueError("Error: Invalid class order file format. Txt: each line a class ID; Json: 1D list.")
        return data
    raise ValueError("Error: Invalid class order file format. Txt: each line a class ID; Json: 1D list.")


def _parse_class_order_arg(class_order: Optional[str], total_classes: int) -> list[int]:
    if class_order is None:
        return []
    if os.path.isfile(class_order):
        order = _load_class_order_from_file(class_order)
    else:
        try:
            parsed = json.loads(class_order)
        except json.JSONDecodeError:
            raise ValueError(
                f"Error: Class order length (?) does not match total classes ({total_classes}). Example of valid order: [0,1,2,...,{total_classes-1}] / Valid txt file: each line is a class ID / Valid json file: [0,1,2,...,{total_classes-1}]"
            )
        if not isinstance(parsed, list) or any(not isinstance(v, int) for v in parsed):
            raise ValueError(
                f"Error: Class order length ({len(parsed) if hasattr(parsed, '__len__') else '?'}) does not match total classes ({total_classes}). Example of valid order: [0,1,2,...,{total_classes-1}] / Valid txt file: each line is a class ID / Valid json file: [0,1,2,...,{total_classes-1}]"
            )
        order = parsed
    if len(order) != total_classes:
        raise ValueError(
            f"Error: Class order length ({len(order)}) does not match total classes ({total_classes}). Example of valid order: [0,1,2,...,{total_classes-1}] / Valid txt file: each line is a class ID / Valid json file: [0,1,2,...,{total_classes-1}]"
        )
    if min(order) < 0 or max(order) >= total_classes:
        raise ValueError(
            f"Error: Class order contains invalid id. All ids must be in [0, {total_classes-1}]."
        )
    if len(set(order)) != total_classes:
        raise ValueError(
            "Error: Class order contains duplicates. Provide a permutation of [0..N-1]."
        )
    return order


def _build_class_order(total_classes: int, class_order: Optional[str], class_order_seed: Optional[int]) -> list[int]:
    if class_order:
        return _parse_class_order_arg(class_order, total_classes)
    order = list(range(total_classes))
    rng = np.random.RandomState(seed=class_order_seed)
    order = rng.permutation(order).tolist()
    return order


def _resolve_task_plan(
    total_classes: int,
    base_classes: Optional[int],
    incremental_classes: Optional[int],
) -> tuple[int, int, int]:
    if incremental_classes is None:
        raise ValueError("Error: --incremental_classes must be provided for class-incremental learning.")
    if incremental_classes <= 0:
        raise ValueError("Error: --incremental_classes must be positive.")

    if base_classes is None:
        base_classes = incremental_classes
    if base_classes <= 0:
        raise ValueError("Error: --base_classes must be positive when provided.")
    if base_classes > total_classes:
        raise ValueError(
            f"Error: base_classes ({base_classes}) exceeds total classes ({total_classes}). Please lower base_classes."
        )

    remaining = total_classes - base_classes
    if remaining < 0:
        raise ValueError("Error: base_classes cannot exceed total classes.")
    if remaining % incremental_classes != 0:
        raise ValueError(
            f"Error: Total classes ({total_classes}) cannot be split with base_classes ({base_classes}) and incremental_classes ({incremental_classes}). Please adjust them to cover all classes evenly."
        )

    extra_tasks = remaining // incremental_classes
    total_tasks = 1 + extra_tasks
    if total_tasks < 2:
        raise ValueError("Error: total_tasks must be ≥2 for class-incremental learning.")
    return total_tasks, base_classes, incremental_classes


def _chunk_classes(class_order: list[int], base_classes: int, incremental_classes: int, total_tasks: int) -> list[list[int]]:
    splits = []
    cursor = 0
    splits.append(class_order[cursor : cursor + base_classes])
    cursor += base_classes
    for _ in range(total_tasks - 1):
        splits.append(class_order[cursor : cursor + incremental_classes])
        cursor += incremental_classes
    return splits


def _ensure_ray():
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)


def main() -> None:
    raise SystemExit(
        "image_cls_cil.py is a library module. Launch RaPO with "
        "python -m examples.baselines.img_cls_cil.image_cls_cil_rapo "
        "or bash scripts/img_cls/*/*.sh"
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
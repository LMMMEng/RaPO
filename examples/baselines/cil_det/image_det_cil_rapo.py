"""image_det_cil_rapo.py
===============================================================================
Class-incremental object detection with RaPO and persistent workers.

This is the main CIL detection entrypoint. Validation reports bbox mAP
as the primary metric; optional SAM mask diagnostics are disabled unless a SAM
checkpoint is explicitly supplied.

Runtime choices:
1. No Ray restart between tasks
2. In-memory weight copy from actor to the frozen previous-task anchor; the KL reference stays the original pretrained model
3. Optimizer state carry-over
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Any, Iterable, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf

# ── Object Detection CIL base module ──────────────────────────────────────────────
import examples.baselines.cil_det.image_det_cil as det_cil_base
from examples.baselines._rapo_components import (
    AnchorRefConfig,
    RetentionRewardConfig,
    EMAAdvConfig,
    EMAAdvNormalizer,
    RayPPOContinualRaPOTrainer,
    _ema_stats_file_from_task_dir,
    _load_ema_state,
    _save_ema_state,
    _load_cil_cfg,
    _build_cil_cfg,
)

from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import ResourcePoolManager, Role
from verl.utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from verl.utils.tokenizer import get_processor, get_tokenizer
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.reward import AutoRewardManager

# ── Worker-base MRO proxy (lets PersistentRefFSDPWorker subclass FSDPWorker) ──
from examples.baselines.cil_misc.fsdp_worker_ext import _FSDPWorkerColocBase


# =========================================================================
# PersistentRefFSDPWorker (mirrors the image_cls variant —
# handles weight management only, no data/metric logic)
# =========================================================================

class PersistentRefFSDPWorker(_FSDPWorkerColocBase, FSDPWorker):
    """FSDPWorker with in-memory weight copy for persistent-worker CIL."""

    def __init__(self, config, role):
        super().__init__(config, role)
        # RaPO always builds a frozen anchor (the previous-task policy) whenever
        # a reference model exists; the actual previous-task weights are copied
        # in-memory before each task from activate_from_task onward.
        if self._has_ref:
            self._has_anchor: bool = True
            t_path = getattr(config.actor.model, "anchor_model_path", None)
            if not t_path or not str(t_path).strip():
                config.actor.model.anchor_model_path = config.actor.model.model_path
        else:
            self._has_anchor = False

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_anchor(self) -> None:
        if not self._has_anchor:
            return
        anchor_model_path: str = getattr(
            self.config.actor.model, "anchor_model_path", ""
        )
        existing_ref = self.ref_fsdp_module
        original_path: str = self.config.actor.model.model_path
        try:
            self.config.actor.model.model_path = anchor_model_path
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )
        finally:
            self.config.actor.model.model_path = original_path
        self.anchor_fsdp_module = self.ref_fsdp_module
        self.ref_fsdp_module = existing_ref

        from verl.workers.actor.dp_actor import DataParallelPPOActor
        self.anchor_policy = DataParallelPPOActor(
            config=self.config.ref,
            actor_module=self.anchor_fsdp_module,
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_anchor_log_probs(self, data: DataProto) -> DataProto:
        # The frozen anchor (previous-task model) is mandatory for the retention
        # reward.  A missing anchor means init_anchor() did not run or the
        # per-task actor->anchor weight copy failed; fail loudly rather than
        # silently substituting the ref model.
        if not self._has_anchor:
            raise RuntimeError(
                "compute_anchor_log_probs() called without an initialised anchor "
                "model. Ensure init_anchor() ran after init_workers() and that the "
                "anchor weights were copied for this task."
            )
        assert hasattr(self, "anchor_fsdp_module")
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        if self._use_ref_param_offload:
            load_fsdp_model(self.anchor_fsdp_module)
        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.anchor_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"anchor_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        if self.world_size > 1:
            self.anchor_fsdp_module._handle.reshard(True)
        if self._use_ref_param_offload:
            offload_fsdp_model(self.anchor_fsdp_module)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def copy_actor_to_anchor(self) -> None:
        if not getattr(self, "_has_anchor", False) or not hasattr(self, "anchor_fsdp_module"):
            raise RuntimeError("copy_actor_to_anchor called without an initialised anchor module")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        if self._use_ref_param_offload:
            load_fsdp_model(self.anchor_fsdp_module)
        with torch.no_grad():
            for p_a, p_t in zip(
                self.fsdp_module.parameters(),
                self.anchor_fsdp_module.parameters(),
            ):
                p_t.data.copy_(p_a.data.to(p_t.dtype))
        import torch.distributed as dist
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        if self._use_ref_param_offload:
            offload_fsdp_model(self.anchor_fsdp_module)

# =========================================================================
# PersistentDetCILTrainer (uses object detection validation: mAP_box + mAP_mask)
# =========================================================================

class PersistentDetCILTrainer(RayPPOContinualRaPOTrainer):
    """Reusable trainer across object detection CIL tasks.

    Inherits EMA-ADV + retention from RayPPOContinualRaPOTrainer.
    Overrides _validate() with object detection mAP evaluation
    (both bbox and mask via SAM).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preset_global_step: Optional[int] = None
        self.sam_checkpoint: Optional[str] = None
        self.sam_model_type: str = "vit_h"

    def _load_checkpoint(self) -> None:
        if self._preset_global_step is not None:
            self.global_step = self._preset_global_step
            print(f"[DET-CIL] Set global_step={self.global_step} (weights already in memory)")
            return
        super()._load_checkpoint()

    def reinit_for_task(
        self,
        train_dataloader,
        val_dataloader,
        *,
        task_id: int,
        per_task_steps: int,
        last_global_step: int,
        save_checkpoint_path: str,
        load_checkpoint_path: Optional[str],
        ema_adv_state: Optional[dict] = None,
        enable_anchor: bool = False,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self._task_id = task_id
        self.config.trainer.save_checkpoint_path = save_checkpoint_path
        self.config.trainer.find_last_checkpoint = False

        if task_id == 1:
            self.config.trainer.load_checkpoint_path = load_checkpoint_path
            self._preset_global_step = None
        else:
            self.config.trainer.load_checkpoint_path = None
            self._preset_global_step = last_global_step

        new_max_steps = per_task_steps + last_global_step
        self.config.trainer.max_steps = new_max_steps
        self.training_steps = new_max_steps
        self.output_dir = save_checkpoint_path

        if self.ema_adv is not None:
            self.ema_adv = EMAAdvNormalizer(
                self.ema_adv_config,
                initial_state=ema_adv_state,
                task_id=task_id,
            )

        self._anchor_available = enable_anchor
        self._pending_retention_metrics = {}
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

    # ------------------------------------------------------------------
    # Object Detection validation (mAP_box + mAP_mask)
    # ------------------------------------------------------------------
    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        reward_metrics_lst = {}
        length_metrics_lst = {}
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        collect_sample_level = bool(getattr(self, "collect_sample_level", False))

        all_pred_dets: list[list[dict]] = []
        all_gt_dets: list[list[dict]] = []
        all_gt_seg_dets: list[list[dict]] = []
        all_image_paths: list[str] = []
        all_image_sizes: list[tuple[int, int]] = []
        sample_gt_cats: list[set[str]] = []

        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
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

            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

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
                batch_seg = test_batch.non_tensor_batch.get("answer_seg")
                batch_multi = test_batch.non_tensor_batch.get("multi_modal_data")

                for bi, (pred_text, gt_str) in enumerate(zip(output_texts, batch_labels)):
                    pred_answer = det_cil_base._extract_answer_text(pred_text)
                    pred_dets = det_cil_base._parse_detections(pred_answer)
                    gt_dets = det_cil_base._parse_detections(str(gt_str))
                    all_pred_dets.append(pred_dets)
                    all_gt_dets.append(gt_dets)
                    sample_gt_cats.append({d["category"] for d in gt_dets})

                    # Seg GT
                    if batch_seg is not None and bi < len(batch_seg):
                        seg_str = batch_seg[bi] if isinstance(batch_seg[bi], str) else str(batch_seg[bi])
                        try:
                            seg_dets = json.loads(seg_str)
                        except (json.JSONDecodeError, TypeError):
                            seg_dets = []
                    else:
                        seg_dets = []
                    all_gt_seg_dets.append(seg_dets)

                    # Image path and size
                    img_path = ""
                    img_w, img_h = 1000, 1000
                    if batch_multi is not None and bi < len(batch_multi):
                        mmd = batch_multi[bi]
                        if isinstance(mmd, dict) and "images" in mmd:
                            imgs = mmd["images"]
                            if isinstance(imgs, list) and imgs:
                                img_path = imgs[0]
                    all_image_paths.append(img_path)

                    img_width_batch = test_batch.non_tensor_batch.get("image_width")
                    img_height_batch = test_batch.non_tensor_batch.get("image_height")
                    if img_width_batch is not None and img_height_batch is not None:
                        try:
                            img_w = int(img_width_batch[bi])
                            img_h = int(img_height_batch[bi])
                        except (IndexError, ValueError, TypeError):
                            pass
                    all_image_sizes.append((img_w, img_h))

            for key, value in self._compute_length_metrics(test_batch).items():
                length_metrics_lst.setdefault(key, []).append(value)

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
                pass

        self.actor_rollout_ref_wg.release_rollout_engine()

        if reward_tensor_lst:
            stacked = torch.cat(reward_tensor_lst, dim=0)
            self.val_reward_score = stacked.sum(-1).mean().item()
        else:
            self.val_reward_score = 0.0

        from verl.trainer.metrics import reduce_metrics
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        print("Finish validation.")

        # Compute bbox-only mAP (always available)
        map_results = det_cil_base._compute_det_map(all_pred_dets, all_gt_dets)
        mAP_box = map_results["mAP"]

        result: dict[str, Any] = {
            "val/reward_score": self.val_reward_score,
            "val/det_mAP_box": mAP_box,
            "val/det_mAP_box_50": map_results.get("mAP_50", mAP_box),
            "val/det_mAP_box_75": map_results.get("mAP_75", 0.0),
            **val_reward_metrics,
            **val_length_metrics,
        }

        # Compute mask mAP via SAM if available
        sam_ckpt = self.sam_checkpoint
        sam_type = self.sam_model_type
        has_seg_gt = any(len(seg) > 0 for seg in all_gt_seg_dets)
        run_sam = getattr(self, "_sam_during_train_val", False)
        per_image_pred_seg = None

        if sam_ckpt and has_seg_gt and run_sam:
            print("  Computing mask mAP via SAM (SAMActor)...")
            try:
                from examples.baselines.cil_det.sam_actor import SAMActor
                sam_handle = SAMActor.remote(sam_ckpt, sam_type)
                mask_results = det_cil_base._compute_object_det_map(
                    all_pred_dets, all_gt_seg_dets,
                    sam_handle, all_image_paths, all_image_sizes,
                    return_per_image=collect_sample_level,
                )
                ray.kill(sam_handle)
                per_image_pred_seg = mask_results.pop("_per_image_pred_seg", None)
                result["val/det_sam_mAP_mask"] = mask_results["mAP_mask"]
                result["val/det_sam_mAP_mask_50"] = mask_results["mAP_mask_50"]
                result["val/det_sam_mAP_mask_75"] = mask_results.get("mAP_mask_75", 0.0)
                # Overwrite box mAP from full object_det eval (uses real image sizes)
                result["val/det_mAP_box"] = mask_results["mAP_box"]
                result["val/det_mAP_box_50"] = mask_results["mAP_box_50"]
                result["val/det_mAP_box_75"] = mask_results.get("mAP_box_75", 0.0)
                print(f"  mAP_box={mask_results['mAP_box']:.4f} mAP_mask={mask_results['mAP_mask']:.4f}")
            except Exception as e:
                print(f"  Warning: mask evaluation failed: {e}")
                result["val/det_sam_mAP_mask"] = 0.0
                result["val/det_sam_mAP_mask_50"] = 0.0
                result["val/det_sam_mAP_mask_75"] = 0.0

        if collect_sample_level:
            result["_all_pred_dets"] = all_pred_dets
            result["_all_gt_dets"] = all_gt_dets
            result["_sample_gt_cats"] = sample_gt_cats
            result["_all_gt_seg_dets"] = all_gt_seg_dets
            result["_all_image_sizes"] = all_image_sizes
            if per_image_pred_seg is not None:
                result["_per_image_pred_seg"] = per_image_pred_seg

        return result

    def _compute_length_metrics(self, test_batch: DataProto) -> dict[str, Any]:
        lengths = {}
        input_lengths = [len(ids) for ids in test_batch.batch["input_ids"]]
        response_lengths = [len(ids) for ids in test_batch.batch["responses"]]
        lengths["input_len"] = sum(input_lengths) / len(input_lengths)
        lengths["response_len"] = sum(response_lengths) / len(response_lengths)
        return lengths


# =========================================================================
# PersistentRunner for object detection CIL
# =========================================================================

@ray.remote(num_cpus=1)
class PersistentRunner:
    """Ray actor that initialises workers once and executes multiple object detection tasks."""

    def __init__(self):
        # Ray sets CUDA_VISIBLE_DEVICES="" for actors with num_gpus=0.
        # Pop it NOW, before any CUDA init, so SAM can use GPU later.
        _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if _cuda_vis is not None and _cuda_vis.strip() == "":
            os.environ.pop("CUDA_VISIBLE_DEVICES")

        self._trainer: Optional[PersistentDetCILTrainer] = None
        self._tokenizer = None
        self._processor = None
        self._reward_fn = None
        self._val_reward_fn = None

    def init(
        self,
        config: PPOConfig,
        categories_path: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_h",
    ) -> None:
        self._tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self._processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self._categories_path = categories_path
        self._sam_checkpoint = sam_checkpoint
        self._sam_model_type = sam_model_type

        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(PersistentRefFSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        remote_reward_manager = ray.remote(AutoRewardManager).options(
            num_cpus=config.worker.reward.num_cpus
        )
        self._reward_fn = remote_reward_manager.remote(config.worker.reward, self._tokenizer)
        self._val_reward_fn = remote_reward_manager.remote(config.worker.reward, self._tokenizer)

        # Build dummy dataloaders for trainer construction
        dummy_train = det_cil_base._build_dataloader(
            config.data, self._tokenizer, self._processor,
            allowed_classes=None, is_train=True,
            categories_path=categories_path,
        )
        dummy_val = det_cil_base._build_dataloader(
            config.data, self._tokenizer, self._processor,
            allowed_classes=None, is_train=False,
            categories_path=categories_path,
        )

        ema_adv_config = EMAAdvConfig.from_env()
        retention_cfg = RetentionRewardConfig.from_env()

        self._trainer = PersistentDetCILTrainer(
            config=config,
            tokenizer=self._tokenizer,
            processor=self._processor,
            train_dataloader=dummy_train,
            val_dataloader=dummy_val,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=self._reward_fn,
            val_reward_fn=self._val_reward_fn,
            ema_adv_config=ema_adv_config,
            ema_adv_state=None,
            retention_cfg=retention_cfg,
            task_id=1,
        )
        self._trainer.init_workers()
        self._trainer.sam_checkpoint = sam_checkpoint
        self._trainer.sam_model_type = sam_model_type
        self._trainer._sam_during_train_val = False
        print("[DET-CIL] Workers initialised (persistent).")

    def run_task(
        self,
        config: PPOConfig,
        task_id: int,
        val_progress: bool,
        val_write_predictions: bool,
        val_output_dir: Optional[str],
        allowed_classes: Optional[Iterable[str]] = None,
        val_allowed_classes: Optional[Iterable[str]] = None,
        allowed_class_order: Optional[list[str]] = None,
        extra_val_splits: Optional[list[dict[str, Any]]] = None,
        prompt_label_list: Optional[list[str]] = None,
        copy_actor_to_anchor: bool = False,
        enable_anchor: bool = False,
        task_id_filter: Optional[int] = None,
        class_to_task: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        trainer = self._trainer
        tokenizer = self._tokenizer
        processor = self._processor
        trainer._task_id = task_id

        if copy_actor_to_anchor:
            print(f"[DET-CIL] task {task_id}: copying actor → anchor")
            trainer.actor_rollout_ref_wg.copy_actor_to_anchor()

        effective_val_allowed_classes = val_allowed_classes if val_allowed_classes is not None else allowed_classes

        # Build task-specific dataloaders (object detection)
        train_dataloader = det_cil_base._build_dataloader(
            config.data, tokenizer, processor, allowed_classes,
            is_train=True, label_list_override=prompt_label_list,
            allowed_class_order=allowed_class_order,
            categories_path=self._categories_path,
            task_id_filter=task_id_filter,
            class_to_task=class_to_task,
        )
        val_dataloader = det_cil_base._build_dataloader(
            config.data, tokenizer, processor, effective_val_allowed_classes,
            is_train=False, label_list_override=prompt_label_list,
            allowed_class_order=allowed_class_order,
            categories_path=self._categories_path,
        )

        per_task_steps = config.trainer.max_steps
        if config.trainer.total_epochs is not None:
            try:
                per_task_steps = config.trainer.total_epochs * len(train_dataloader)
            except Exception:
                pass

        last_global_step = 0
        if config.trainer.load_checkpoint_path:
            try:
                last_global_step = int(
                    os.path.basename(config.trainer.load_checkpoint_path).split("global_step_")[-1]
                )
            except Exception:
                pass

        ema_state_path = _ema_stats_file_from_task_dir(config.trainer.save_checkpoint_path)
        ema_initial_state = _load_ema_state(ema_state_path)

        trainer.reinit_for_task(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            task_id=task_id,
            per_task_steps=per_task_steps,
            last_global_step=last_global_step,
            save_checkpoint_path=config.trainer.save_checkpoint_path,
            load_checkpoint_path=config.trainer.load_checkpoint_path if task_id == 1 else None,
            ema_adv_state=ema_initial_state,
            enable_anchor=enable_anchor,
        )
        trainer.val_progress = val_progress
        trainer.val_write_predictions = val_write_predictions
        trainer.val_output_dir = val_output_dir

        # Skip mid-training validation; CIL does comprehensive post-training eval
        config.trainer.val_freq = -1

        if hasattr(trainer, "output_dir") and trainer.output_dir:
            det_cil_base._snapshot_training_config(trainer.output_dir, config)

        # Train
        trainer.fit()

        # Save EMA state
        if trainer.ema_adv is not None:
            ema_payload = trainer.ema_adv.state_dict()
            _save_ema_state(ema_state_path, ema_payload, task_id=task_id)
            print(f"[EMA-ADV] Saved EMA online state to: {ema_state_path}")

        # Post-training evaluation with extra_val_splits
        extra_eval_results = {}
        if extra_val_splits:
            split_map = {split["name"]: split for split in extra_val_splits}
            overall_split = split_map.get("overall_seen")

            if overall_split is not None:
                overall_val_loader = det_cil_base._build_dataloader(
                    config.data, tokenizer, processor,
                    allowed_classes=overall_split.get("allowed_classes", []),
                    is_train=False,
                    allowed_class_order=overall_split.get("class_order"),
                    categories_path=self._categories_path,
                )
                trainer.val_dataloader = overall_val_loader
                trainer.val_progress = False
                trainer.val_write_predictions = False
                trainer.collect_sample_level = True
                trainer._sam_during_train_val = True
                overall_metrics = trainer._validate()
                trainer._sam_during_train_val = False
                trainer.collect_sample_level = False

                all_pred = overall_metrics.pop("_all_pred_dets", [])
                all_gt = overall_metrics.pop("_all_gt_dets", [])
                sample_gt_cats = overall_metrics.pop("_sample_gt_cats", [])
                all_gt_seg = overall_metrics.pop("_all_gt_seg_dets", None)
                all_img_sizes = overall_metrics.pop("_all_image_sizes", None)
                per_image_masks = overall_metrics.pop("_per_image_pred_seg", None)

                extra_eval_results["overall_seen"] = overall_metrics

                # Per-task mAP from the same overall evaluation pass
                for split in extra_val_splits:
                    split_name = split["name"]
                    if split_name == "overall_seen":
                        continue
                    split_allowed = {
                        det_cil_base._normalize_label_name(str(x))
                        for x in split.get("allowed_classes", [])
                    }
                    split_pred = []
                    split_gt = []
                    for preds, gts, gt_cats in zip(all_pred, all_gt, sample_gt_cats):
                        if not gt_cats.intersection(split_allowed):
                            continue
                        split_pred.append([d for d in preds if d["category"] in split_allowed])
                        split_gt.append([d for d in gts if d["category"] in split_allowed])
                    if split_gt:
                        split_map_result = det_cil_base._compute_det_map(split_pred, split_gt)
                        split_mAP = split_map_result["mAP"]
                    else:
                        split_mAP = 0.0

                    # Per-task mask mAP from cached SAM predictions
                    split_mask_mAP = 0.0
                    if per_image_masks is not None and all_gt_seg is not None:
                        split_mask_mAP = det_cil_base._compute_split_mask_map(
                            per_image_masks, all_gt_seg, all_img_sizes,
                            split_allowed, sample_gt_cats,
                        )

                    extra_eval_results[split_name] = {
                        "val/det_mAP_box": split_mAP,
                        "val/det_sam_mAP_mask": split_mask_mAP,
                    }

        tracker_path = os.path.join(
            config.trainer.save_checkpoint_path, "checkpoint_tracker.json"
        )
        last_checkpoint = None
        if os.path.exists(tracker_path):
            with open(tracker_path, encoding="utf-8") as f:
                tracker_info = json.load(f)
            step = tracker_info.get("last_global_step")
            if step is not None:
                last_checkpoint = os.path.join(
                    config.trainer.save_checkpoint_path, f"global_step_{step}"
                )

        return {
            "extra_eval_results": extra_eval_results,
            "last_checkpoint_path": last_checkpoint,
        }

    def init_anchor_on_workers(self) -> None:
        self._trainer.actor_rollout_ref_wg.init_anchor()


# =========================================================================
# Resolve ref/anchor weight-copy actions (mirrors the image_cls variant)
# =========================================================================

def _resolve_weight_actions(
    ref_cfg: AnchorRefConfig,
    task_id: int,
) -> dict[str, bool]:
    if task_id < ref_cfg.activate_from_task:
        return {
            "copy_actor_to_anchor": False,
            "enable_anchor": False,
        }
    return {
        "copy_actor_to_anchor": True,
        "enable_anchor": True,
    }


# =========================================================================
# Main CIL loop (persistent workers, object detection)
# =========================================================================

def _run_cil(
    ppo_config: PPOConfig,
    known_args: argparse.Namespace,
    ref_cfg: AnchorRefConfig,
    cil_config: Optional[dict] = None,
) -> None:
    import shutil

    current_time = time.strftime("%Y-%m-%d-%H%M%S")
    save_dir = known_args.save_dir
    if save_dir is None:
        raise ValueError("Error: --save_dir is required.")
    save_dir = f"{save_dir}/{current_time}"

    os.makedirs(save_dir, exist_ok=True)
    cil_info_dir = os.path.join(save_dir, "cil_info")
    metrics_dir = os.path.join(cil_info_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    det_cil_base._snapshot_launch_script(cil_info_dir, explicit_script_path=known_args.launch_script)
    det_cil_base._snapshot_repro_files(cil_info_dir, ppo_config)

    if cil_config is not None:
        with open(os.path.join(cil_info_dir, "cil_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(cil_config, f, indent=2, ensure_ascii=False)

    # Load categories
    if known_args.categories:
        class_names = det_cil_base._load_categories(known_args.categories)
    else:
        class_names = det_cil_base._list_class_names_from_jsonl(ppo_config.data.train_files)
    class_names = [det_cil_base._normalize_label_name(c) for c in class_names]
    total_classes = len(class_names)

    total_tasks, base_classes, incremental_classes = det_cil_base._resolve_task_plan(
        total_classes, known_args.base_classes, known_args.incremental_classes
    )
    class_order = det_cil_base._build_class_order(total_classes, known_args.class_order, known_args.class_order_seed)
    class_splits = det_cil_base._chunk_classes(class_order, base_classes, incremental_classes, total_tasks)
    class_order_names = [class_names[i] for i in class_order]
    class_to_task = det_cil_base._build_class_to_task_map(class_splits, class_names)

    with open(os.path.join(cil_info_dir, "class_order.json"), "w", encoding="utf-8") as f:
        json.dump({
            "class_order_seed": known_args.class_order_seed,
            "final_class_order_ids": class_order,
            "final_class_order_names": class_order_names,
        }, f, indent=2)

    all_labels_path = os.path.join(cil_info_dir, "all_labels.json")
    with open(all_labels_path, "w", encoding="utf-8") as f:
        json.dump(class_order_names, f, ensure_ascii=True)

    # Precompute per-task sample counts
    task_samples_info: list[dict[str, Any]] = []
    for idx, ids in enumerate(class_splits):
        ordered_class_names = [class_names[i] for i in ids]
        allowed_norm = {det_cil_base._normalize_label_name(name) for name in ordered_class_names}
        train_samples = det_cil_base._count_task_images(ppo_config.data.train_files, class_to_task, idx)
        val_samples = det_cil_base._count_class_images(ppo_config.data.val_files, allowed_norm)
        task_samples_info.append({
            "task": idx + 1,
            "classes": ordered_class_names,
            "train_samples": train_samples,
            "val_samples": val_samples,
        })

    with open(os.path.join(cil_info_dir, "task_samples.json"), "w", encoding="utf-8") as f:
        json.dump(task_samples_info, f, indent=2)

    # CIL metrics tracking (mAP-based, both box and mask)
    overall_last_mAP_box_history: list[float] = []
    overall_last_mAP_mask_history: list[float] = []
    summary: dict[str, Any] = {
        "class_order_seed": known_args.class_order_seed,
        "final_class_order_ids": class_order,
        "final_class_order_names": class_order_names,
        "total_tasks": total_tasks,
        "base_classes": base_classes,
        "incremental_classes": incremental_classes,
        "tasks": {},
    }

    last_checkpoint: Optional[str] = ppo_config.trainer.load_checkpoint_path
    previous_task_dir: Optional[str] = None
    seen_class_ids: list[int] = []
    task_mAP_matrix: list[list[float]] = []
    task_mAP_mask_matrix: list[list[float]] = []

    # Wandb
    wandb_available = False
    if os.environ.get("WANDB_DISABLED", "false").lower() not in {"true", "1", "yes"}:
        try:
            import wandb
            shared_run_id = wandb.util.generate_id()
            shared_run_name = f"{ppo_config.trainer.experiment_name}-inst-seg-cil-{time.strftime('%Y%m%d-%H%M%S')}"
            os.environ["WANDB_RUN_ID"] = shared_run_id
            os.environ.setdefault("WANDB_RESUME", "allow")
            os.environ.setdefault("WANDB_RUN_NAME", shared_run_name)
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "120")
            wandb_available = True
        except Exception as exc:
            print(f"Warning: wandb setup failed: {exc}")

    # Ensure Ray and create persistent runner
    det_cil_base._ensure_ray()
    runner = PersistentRunner.remote()
    ray.get(runner.init.remote(
        ppo_config,
        categories_path=known_args.categories,
        sam_checkpoint=None if known_args.det_only else known_args.sam_checkpoint,
        sam_model_type=known_args.sam_model_type,
    ))

    # Initialise the frozen anchor model on all workers. A failure here is fatal:
    # the retention reward requires anchor log-probs from task 2 onward.
    ray.get(runner.init_anchor_on_workers.remote())

    print(f"[DET-CIL] Starting {total_tasks} object detection CIL tasks with persistent workers.")

    # Task loop
    for task_idx, task_class_ids in enumerate(class_splits):
        task_id = task_idx + 1
        seen_class_ids.extend(task_class_ids)
        ordered_task_class_names = [class_names[i] for i in task_class_ids]
        prompt_label_list = None
        if known_args.prompt_seen_labels:
            prompt_label_list = [class_names[i] for i in seen_class_ids]

        task_config = copy.deepcopy(ppo_config)
        task_save_dir = os.path.join(save_dir, f"task_{task_id}")
        os.makedirs(task_save_dir, exist_ok=True)
        task_config.trainer.save_checkpoint_path = task_save_dir
        task_config.trainer.find_last_checkpoint = False
        task_config.trainer.skip_dataloader_state = task_idx > 0
        task_config.trainer.load_checkpoint_path = (
            last_checkpoint if task_idx > 0
            else ppo_config.trainer.load_checkpoint_path
        )

        seen_class_names = [class_names[i] for i in seen_class_ids]
        eval_plan = []
        for eval_idx, eval_class_ids in enumerate(class_splits[:task_idx + 1]):
            ordered_eval_classes = [class_names[i] for i in eval_class_ids]
            eval_plan.append({
                "name": f"task{eval_idx + 1}",
                "allowed_classes": ordered_eval_classes,
                "class_order": seen_class_names,
            })
        eval_plan.append({
            "name": "overall_seen",
            "allowed_classes": seen_class_names,
            "class_order": seen_class_names,
        })

        weight_actions = _resolve_weight_actions(ref_cfg, task_id)

        result = ray.get(
            runner.run_task.remote(
                config=task_config,
                task_id=task_id,
                val_progress=not known_args.no_val_progress,
                val_write_predictions=known_args.val_write_predictions,
                val_output_dir=known_args.val_output_dir,
                allowed_classes=ordered_task_class_names,
                val_allowed_classes=seen_class_names,
                allowed_class_order=prompt_label_list or ordered_task_class_names,
                extra_val_splits=eval_plan,
                prompt_label_list=prompt_label_list,
                copy_actor_to_anchor=weight_actions["copy_actor_to_anchor"],
                enable_anchor=weight_actions["enable_anchor"],
                task_id_filter=task_idx,
                class_to_task=class_to_task,
            )
        )

        extra_eval_results = result.get("extra_eval_results", {}) if isinstance(result, dict) else {}
        last_checkpoint = result.get("last_checkpoint_path") if isinstance(result, dict) else None

        # CIL metrics (mAP-based, both box and mask)
        overall_metrics = extra_eval_results.get("overall_seen", {})
        overall_last_mAP_box = overall_metrics.get("val/det_mAP_box", 0.0)
        overall_last_mAP_mask = overall_metrics.get("val/det_sam_mAP_mask", 0.0)
        overall_last_mAP_box_history.append(overall_last_mAP_box)
        overall_last_mAP_mask_history.append(overall_last_mAP_mask)
        cumulative_avg_box = sum(overall_last_mAP_box_history) / len(overall_last_mAP_box_history)
        cumulative_avg_mask = sum(overall_last_mAP_mask_history) / len(overall_last_mAP_mask_history)

        per_task_mAP_raw = {
            f"task{j}": extra_eval_results.get(f"task{j}", {}).get("val/det_mAP_box", 0.0)
            for j in range(1, task_id + 1)
        }
        per_task_mAP_pct = {k: round(v * 100, 4) for k, v in per_task_mAP_raw.items()}

        per_task_mAP_mask_raw = {
            f"task{j}": extra_eval_results.get(f"task{j}", {}).get("val/det_sam_mAP_mask", 0.0)
            for j in range(1, task_id + 1)
        }
        per_task_mAP_mask_pct = {k: round(v * 100, 4) for k, v in per_task_mAP_mask_raw.items()}

        current_row = [per_task_mAP_raw[f"task{j}"] for j in range(1, task_id + 1)]
        task_mAP_matrix.append(current_row)
        current_row_mask = [per_task_mAP_mask_raw[f"task{j}"] for j in range(1, task_id + 1)]
        task_mAP_mask_matrix.append(current_row_mask)

        fr_box: Optional[float] = None
        fr_mask: Optional[float] = None
        if task_id >= 2:
            np_table = np.zeros([task_id, task_id], dtype=np.float32)
            for idxx, line in enumerate(task_mAP_matrix):
                idxy = len(line)
                np_table[idxx, :idxy] = np.array(line, dtype=np.float32)
            np_table = np_table.T
            fr_box = float(np.mean((np.max(np_table, axis=1) - np_table[:, task_id - 1])[:task_id - 1]))

            np_table_mask = np.zeros([task_id, task_id], dtype=np.float32)
            for idxx, line in enumerate(task_mAP_mask_matrix):
                idxy = len(line)
                np_table_mask[idxx, :idxy] = np.array(line, dtype=np.float32)
            np_table_mask = np_table_mask.T
            fr_mask = float(np.mean((np.max(np_table_mask, axis=1) - np_table_mask[:, task_id - 1])[:task_id - 1]))

        def _pct(v: Optional[float]) -> Optional[float]:
            return round(v * 100, 4) if v is not None else None

        summary["tasks"][f"task{task_id}"] = {
            "last_mAP_box": _pct(overall_last_mAP_box),
            "last_mAP_mask": _pct(overall_last_mAP_mask),
            "average_mAP_box": _pct(cumulative_avg_box),
            "average_mAP_mask": _pct(cumulative_avg_mask),
            "last_mAP_box_details": per_task_mAP_pct,
            "last_mAP_mask_details": per_task_mAP_mask_pct,
            "forgetting_rate_box": _pct(fr_box),
            "forgetting_rate_mask": _pct(fr_mask),
        }

        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        task_log_path = os.path.join(metrics_dir, "task_metrics.log")
        with open(task_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task": task_id,
                "last_mAP_box": _pct(overall_last_mAP_box),
                "last_mAP_mask": _pct(overall_last_mAP_mask),
                "average_mAP_box": _pct(cumulative_avg_box),
                "average_mAP_mask": _pct(cumulative_avg_mask),
                "last_mAP_box_details": per_task_mAP_pct,
                "last_mAP_mask_details": per_task_mAP_mask_pct,
                "forgetting_rate_box": _pct(fr_box),
                "forgetting_rate_mask": _pct(fr_mask),
            }) + "\n")

        decomp_parts = [f"T{j}:{per_task_mAP_pct[f'task{j}']:.2f}%" for j in range(1, task_id + 1)]
        extra_str = ""
        if fr_box is not None:
            extra_str += f" | FR_box: {_pct(fr_box):.2f}%"
        if fr_mask is not None:
            extra_str += f" | FR_mask: {_pct(fr_mask):.2f}%"
        mask_str = f" | mAP_mask: {_pct(overall_last_mAP_mask):.2f}%" if overall_last_mAP_mask else ""
        print(
            f"[Task {task_id} Finished] "
            f"Last mAP_box: {_pct(overall_last_mAP_box):.2f}%{mask_str}"
            f" | Avg_box: {_pct(cumulative_avg_box):.2f}% Avg_mask: {_pct(cumulative_avg_mask):.2f}%"
            f" | Decomp: {' '.join(decomp_parts)}"
            f"{extra_str}"
        )

        # Checkpoint pruning
        if known_args.save_task_ckpt in ("latest", "none") and last_checkpoint:
            try:
                ckpt_dirname = os.path.basename(last_checkpoint.rstrip(os.sep))
                for entry in os.listdir(task_save_dir):
                    full_path = os.path.join(task_save_dir, entry)
                    if os.path.isdir(full_path) and entry.startswith("global_step_") and entry != ckpt_dirname:
                        shutil.rmtree(full_path)
            except Exception as exc:
                print(f"Warning: checkpoint prune failed in {task_save_dir}: {exc}")

        if known_args.save_task_ckpt in ("latest", "none") and previous_task_dir and previous_task_dir != task_save_dir:
            try:
                for entry in os.listdir(previous_task_dir):
                    full_path = os.path.join(previous_task_dir, entry)
                    if os.path.isdir(full_path) and entry.startswith("global_step_"):
                        shutil.rmtree(full_path)
            except Exception as exc:
                print(f"Warning: checkpoint prune failed in {previous_task_dir}: {exc}")

        previous_task_dir = task_save_dir

        if wandb_available:
            try:
                import wandb
                if wandb.run is not None:
                    wandb_payload: dict[str, Any] = {
                        "cil/task": task_id,
                        "cil/overall_last_mAP_box": _pct(overall_last_mAP_box),
                        "cil/overall_last_mAP_mask": _pct(overall_last_mAP_mask),
                        "cil/cumulative_average_mAP_box": _pct(cumulative_avg_box),
                        "cil/cumulative_average_mAP_mask": _pct(cumulative_avg_mask),
                        "cil/train_samples": task_samples_info[task_idx]["train_samples"],
                        "cil/val_samples": task_samples_info[task_idx]["val_samples"],
                        **({"cil/forgetting_rate_box": _pct(fr_box)} if fr_box is not None else {}),
                        **({"cil/forgetting_rate_mask": _pct(fr_mask)} if fr_mask is not None else {}),
                    }
                    wandb.log(wandb_payload)
            except Exception as exc:
                print(f"Warning: wandb log failed at task {task_id}: {exc}")

    if known_args.save_task_ckpt == "none" and previous_task_dir:
        try:
            for entry in os.listdir(previous_task_dir):
                full_path = os.path.join(previous_task_dir, entry)
                if os.path.isdir(full_path) and entry.startswith("global_step_"):
                    shutil.rmtree(full_path)
        except Exception as exc:
            print(f"Warning: checkpoint prune failed in final task dir {previous_task_dir}: {exc}")

    ray.kill(runner)
    print(f"[DET-CIL] All {total_tasks} object detection tasks completed.")


# =========================================================================
# Argument parsing
# =========================================================================

def _parse_args() -> "tuple[argparse.Namespace, list[str]]":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cil_cfg", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    cfg_defaults: dict[str, Any] = {}
    if pre_args.cil_cfg is not None:
        cfg_defaults = _load_cil_cfg(pre_args.cil_cfg)
        print(f"[CIL-CFG] Loaded defaults from {pre_args.cil_cfg}")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cil_cfg", type=str, default=None)

    # Standard CIL args
    parser.add_argument("--val_write_predictions", action="store_true")
    parser.add_argument("--val_output_dir", default=None)
    parser.add_argument("--no_val_progress", action="store_true")
    parser.add_argument("--class_order_seed", type=int, default=None)
    parser.add_argument("--class_order", type=str, default=None)
    parser.add_argument("--save_task_ckpt", type=str, default="latest", choices=["none", "latest", "all"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--base_classes", type=int, default=None)
    parser.add_argument("--incremental_classes", type=int, default=None)
    parser.add_argument("--launch_script", type=str, default=None)
    parser.add_argument("--prompt_seen_labels", action="store_true")
    parser.add_argument(
        "--categories", type=str, default=None,
        help="Path to categories.json with ordered class list.",
    )

    # SAM args (object detection specific)
    parser.add_argument("--sam_checkpoint", type=str, default=None,
                        help="Path to SAM checkpoint (e.g. sam_vit_h_4b8939.pth)")
    parser.add_argument("--sam_model_type", type=str, default="vit_h",
                        help="SAM model type: vit_h, vit_l, vit_b")
    parser.add_argument("--det_only", action="store_true",
                        help="Detection-only mode: skip SAM mask evaluation even if sam_checkpoint is set.")

    # CTAN args
    parser.add_argument("--ctan_enable", dest="ema_adv_enable", action="store_true")
    parser.add_argument("--ctan_beta", dest="ema_adv_beta", type=float, default=0.999)
    parser.add_argument("--ctan_eps", dest="ema_adv_eps", type=float, default=1e-6)
    parser.add_argument("--ctan_bootstrap_steps", dest="ema_adv_bootstrap_steps", type=int, default=0)
    parser.add_argument("--ctan_activate_from_task", dest="ema_adv_activate_from_task", type=int, default=2)
    parser.add_argument("--ctan_min_std", dest="ema_adv_min_std", type=float, default=1e-3)
    parser.add_argument("--ctan_guard_abs_max", dest="ema_adv_guard_abs_max", type=float, default=5.0)
    parser.add_argument("--ctan_bias_correction", dest="ema_adv_bias_correction", action="store_true", default=True)
    parser.add_argument("--no_ctan_bias_correction", dest="ema_adv_bias_correction", action="store_false")
    parser.add_argument("--ctan_beta_warmup_steps", dest="ema_adv_beta_warmup_steps", type=int, default=2)
    parser.add_argument("--ctan_beta_warmup_init", dest="ema_adv_beta_warmup_init", type=float, default=0.9)

    # Retention reward args
    parser.add_argument("--retention_reward_enable", action="store_true", default=False)
    parser.add_argument("--retention_lambda", type=float, default=0.5)
    parser.add_argument("--retention_alpha", type=float, default=20.0)
    parser.add_argument("--retention_activate_from_task", type=int, default=2)

    # ref = frozen original pretrained; anchor = frozen previous-task checkpoint.
    parser.add_argument("--rapo_activate_from_task", type=int, default=2)

    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    return parser.parse_known_args()


def _set_env(known_args: argparse.Namespace) -> None:
    os.environ["EMA_ADV_ENABLED"] = "1" if known_args.ema_adv_enable else "0"
    os.environ["EMA_ADV_BETA"] = str(known_args.ema_adv_beta)
    os.environ["EMA_ADV_EPS"] = str(known_args.ema_adv_eps)
    os.environ["EMA_ADV_BOOTSTRAP_STEPS"] = str(known_args.ema_adv_bootstrap_steps)
    os.environ["EMA_ADV_ACTIVATE_FROM_TASK"] = str(known_args.ema_adv_activate_from_task)
    os.environ["EMA_ADV_MIN_STD"] = str(known_args.ema_adv_min_std)
    os.environ["EMA_ADV_GUARD_ABS_MAX"] = str(known_args.ema_adv_guard_abs_max)
    os.environ["EMA_ADV_BIAS_CORRECTION"] = "1" if known_args.ema_adv_bias_correction else "0"
    os.environ["EMA_ADV_BETA_WARMUP_STEPS"] = str(known_args.ema_adv_beta_warmup_steps)
    os.environ["EMA_ADV_BETA_WARMUP_INIT"] = str(known_args.ema_adv_beta_warmup_init)
    os.environ["RETENTION_REWARD_ENABLED"] = "1" if known_args.retention_reward_enable else "0"
    os.environ["RETENTION_REWARD_LAMBDA"] = str(known_args.retention_lambda)
    os.environ["RETENTION_ALPHA"] = str(known_args.retention_alpha)
    os.environ["RETENTION_REWARD_ACTIVATE_FROM_TASK"] = str(known_args.retention_activate_from_task)


# =========================================================================
# Entry point
# =========================================================================

def main() -> None:
    known_args, remaining = _parse_args()
    _set_env(known_args)

    cli_args = OmegaConf.from_cli(remaining)
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    ref_cfg = AnchorRefConfig(
        activate_from_task=known_args.rapo_activate_from_task,
    )

    print(
        "[DET-CIL] Configuration:\n"
        f"  reference=original pretrained base (frozen KL anchor)\n"
        f"  anchor=previous-task policy (frozen retention target)\n"
        f"  activate_from_task={ref_cfg.activate_from_task}\n"
        f"  sam_checkpoint={known_args.sam_checkpoint}\n"
        f"  sam_model_type={known_args.sam_model_type}\n"
        f"  original_model_path={ppo_config.worker.actor.model.model_path}"
    )

    cil_config = _build_cil_cfg(known_args)
    _run_cil(ppo_config, known_args, ref_cfg, cil_config=cil_config)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

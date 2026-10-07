"""image_cls_cil_rapo.py
===============================================================================
Class-incremental image classification with RaPO and persistent workers.

Runtime choices:

1. **No Ray restart between tasks** – Worker processes (FSDP + vLLM) are
   spawned once and reused across all CIL tasks, eliminating the ~2-3 min
   per-task overhead of process creation + NCCL init + model loading.

2. **In-memory weight copy instead of HF merge** – From task 2 onward the
   actor FSDP weights are copied into the frozen **anchor** module (previous
   task). The KL **reference** stays the original pretrained weights.

3. **Optimizer state carry-over** – Between tasks the actor's optimizer state
   (AdamW momentum, variance, step counter) is kept in memory, matching the
   original code where ``FSDPCheckpointManager.load_checkpoint()`` restores
   the full optimizer state from the previous task's checkpoint.

CIL evaluation (last_acc, avg_acc, forgetting_rate, per-task decomposition,
extra_val_splits with overall_seen, sample-level metrics) is implemented in
``image_cls_cil.py`` and reused here.
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

# ── Base CIL module and RaPO components ─────────────────────────────────────
# We import heavily from the existing modules to avoid duplicating code.
import examples.baselines.img_cls_cil.image_cls_cil as base
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
# PersistentRefFSDPWorker – FSDPWorker with in-memory weight copy across models
# =========================================================================

class PersistentRefFSDPWorker(_FSDPWorkerColocBase, FSDPWorker):
    """FSDPWorker that supports in-memory weight copy across FSDP models.

    Adds methods for:
    - Copying actor weights → anchor (previous-task, from task 2 onward)
    - Initialising and running the frozen anchor model

    Uses ``_FSDPWorkerColocBase`` as the first parent so that verl's colocated
    worker construction still works (see ``fsdp_worker_ext``).
    """

    def __init__(self, config, role):
        super().__init__(config, role)
        # RaPO always builds a frozen anchor (the previous-task policy) whenever
        # a reference model exists. init_anchor() loads the anchor module from
        # model_path (the original weights at init time); the actual previous-
        # task weights are copied in-memory before each task from
        # activate_from_task onward.
        if self._has_ref:
            self._has_anchor: bool = True
            t_path = getattr(config.actor.model, "anchor_model_path", None)
            if not t_path or not str(t_path).strip():
                config.actor.model.anchor_model_path = config.actor.model.model_path
        else:
            self._has_anchor = False

    # ------------------------------------------------------------------
    # Anchor init
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Anchor log-probs
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # In-memory FSDP weight copy: actor → anchor
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def copy_actor_to_anchor(self) -> None:
        """Copy actor FSDP flat-param weights to the anchor FSDP module."""
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
# PersistentCILTrainer – reusable trainer across tasks
# =========================================================================

class PersistentCILTrainer(RayPPOContinualRaPOTrainer):
    """Trainer that can re-run fit() multiple times on the same worker group.

    Supports:
    - Swapping train/val dataloaders between tasks
    - Resetting global_step to pick up from last checkpoint step
    - Setting max_steps for each task
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # When set, _load_checkpoint will use this instead of loading from disk.
        self._preset_global_step: Optional[int] = None

    def _load_checkpoint(self) -> None:
        """Override to support in-memory weight persistence.

        For task 1, behaves normally (loads from disk if path set).
        For task 2+, weights are already in memory so we only set global_step.
        """
        if self._preset_global_step is not None:
            self.global_step = self._preset_global_step
            print(f"[CIL] Set global_step={self.global_step} (weights already in memory)")
            return
        # Task 1: normal checkpoint loading
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
        """Re-initialise task-dependent state for a new CIL task."""
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self._task_id = task_id

        # Update config paths
        self.config.trainer.save_checkpoint_path = save_checkpoint_path
        self.config.trainer.find_last_checkpoint = False

        # For task 1, use normal checkpoint loading from disk.
        # For task 2+, weights are already in memory; just set step counter.
        if task_id == 1:
            self.config.trainer.load_checkpoint_path = load_checkpoint_path
            self._preset_global_step = None
        else:
            self.config.trainer.load_checkpoint_path = None
            self._preset_global_step = last_global_step

        # Step management
        new_max_steps = per_task_steps + last_global_step
        self.config.trainer.max_steps = new_max_steps
        self.training_steps = new_max_steps

        # Output dir for snapshots
        self.output_dir = save_checkpoint_path

        # Update EMA state
        if self.ema_adv is not None:
            self.ema_adv = EMAAdvNormalizer(
                self.ema_adv_config,
                initial_state=ema_adv_state,
                task_id=task_id,
            )

        # In persistent mode, auxiliary anchor availability is determined by the
        # enable_anchor flag (derived from ref_cfg + task_id) rather than
        # config paths (which are never modified by hooks in persistent mode).
        self._anchor_available = enable_anchor

        # Clear pending retention metrics from previous task to prevent
        # stale metrics leaking into the next task's first log entry.
        self._pending_retention_metrics = {}

        # Reset best score tracking (must match RayPPOTrainer.__init__ defaults)
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None


# =========================================================================
# Runner – persistent Ray actor that holds workers across all tasks
# =========================================================================

@ray.remote(num_cpus=1)
class PersistentRunner:
    """Ray actor that initialises workers once and executes multiple tasks."""

    def __init__(self):
        self._trainer: Optional[PersistentCILTrainer] = None
        self._tokenizer = None
        self._processor = None
        self._reward_fn = None
        self._val_reward_fn = None

    def init(self, config: PPOConfig) -> None:
        """Initialise workers, tokenizer, processor, reward fns (once)."""
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

        # Build dummy dataloaders for trainer construction (will be replaced per task)
        dummy_train = base._build_dataloader(
            config.data, self._tokenizer, self._processor,
            allowed_classes=None, is_train=True,
        )
        dummy_val = base._build_dataloader(
            config.data, self._tokenizer, self._processor,
            allowed_classes=None, is_train=False,
        )

        ema_adv_config = EMAAdvConfig.from_env()
        retention_cfg = RetentionRewardConfig.from_env()

        self._trainer = PersistentCILTrainer(
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
        print("[CIL] Workers initialised (persistent).")

    def run_task(
        self,
        config: PPOConfig,
        task_id: int,
        val_progress: bool,
        val_write_predictions: bool,
        val_output_dir: Optional[str],
        allowed_classes: Optional[Iterable[str]] = None,
        allowed_class_order: Optional[list[str]] = None,
        extra_val_splits: Optional[list[dict[str, Any]]] = None,
        prompt_label_list: Optional[list[str]] = None,
        copy_actor_to_anchor: bool = False,
        enable_anchor: bool = False,
    ) -> dict[str, Any]:
        """Run a single CIL task using the persistent workers."""
        trainer = self._trainer
        tokenizer = self._tokenizer
        processor = self._processor
        trainer._task_id = task_id

        if copy_actor_to_anchor:
            print(f"[CIL] task {task_id}: copying actor → anchor")
            trainer.actor_rollout_ref_wg.copy_actor_to_anchor()

        # ── Optimizer state carries over between tasks ──────────────────
        # In the original code, FSDPCheckpointManager.load_checkpoint()
        # restores the previous task's full optimizer state (AdamW momentum,
        # variance, step counter) from the checkpoint.  Persistent workers
        # already keep this state in memory, so we do NOT reset it.

        # ── Build task-specific dataloaders ─────────────────────────────
        train_dataloader = base._build_dataloader(
            config.data, tokenizer, processor, allowed_classes,
            is_train=True, label_list_override=prompt_label_list,
            allowed_class_order=allowed_class_order,
        )
        val_dataloader = base._build_dataloader(
            config.data, tokenizer, processor, allowed_classes,
            is_train=False, label_list_override=prompt_label_list,
            allowed_class_order=allowed_class_order,
        )

        # ── Compute per-task steps ──────────────────────────────────────
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

        # ── EMA state ───────────────────────────────────────────────────
        ema_state_path = _ema_stats_file_from_task_dir(config.trainer.save_checkpoint_path)
        ema_initial_state = _load_ema_state(ema_state_path)

        # ── Reinitialise trainer for this task ──────────────────────────
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

        if hasattr(trainer, "output_dir") and trainer.output_dir:
            base._snapshot_training_config(trainer.output_dir, config)

        # ── Train ───────────────────────────────────────────────────────
        trainer.fit()

        # ── Save EMA state ──────────────────────────────────────────────
        if trainer.ema_adv is not None:
            ema_payload = trainer.ema_adv.state_dict()
            _save_ema_state(ema_state_path, ema_payload, task_id=task_id)
            print(f"[EMA-ADV] Saved EMA online state to: {ema_state_path}")

        # ── Post-training evaluation (extra_val_splits) ─────────────────
        extra_eval_results = {}
        if extra_val_splits:
            split_map = {split["name"]: split for split in extra_val_splits}
            overall_split = split_map.get("overall_seen")

            if overall_split is not None:
                overall_val_loader = base._build_dataloader(
                    config.data, tokenizer, processor,
                    allowed_classes=overall_split.get("allowed_classes", []),
                    is_train=False,
                    allowed_class_order=overall_split.get("class_order"),
                )
                trainer.val_dataloader = overall_val_loader
                trainer.val_progress = False
                trainer.val_write_predictions = False
                trainer.collect_sample_level = True
                overall_metrics = trainer._validate()
                trainer.collect_sample_level = False

                sample_acc = [float(x) for x in overall_metrics.pop("_sample_accuracy_scores", [])]
                sample_labels = [str(x) for x in overall_metrics.pop("_sample_labels_norm", [])]
                if len(sample_acc) != len(sample_labels):
                    raise RuntimeError("Sample-level validation trace length mismatch.")

                extra_eval_results["overall_seen"] = overall_metrics

                for split in extra_val_splits:
                    split_name = split["name"]
                    if split_name == "overall_seen":
                        continue
                    split_allowed = {
                        base._normalize_label_name(str(x))
                        for x in split.get("allowed_classes", [])
                    }
                    hit_acc = [
                        a for a, lbl in zip(sample_acc, sample_labels)
                        if lbl in split_allowed
                    ]
                    split_acc = float(np.mean(hit_acc)) if hit_acc else 0.0
                    extra_eval_results[split_name] = {
                        "val/cls_accuracy": split_acc,
                        "val/accuracy_reward": split_acc,
                    }

        # ── Checkpoint path ─────────────────────────────────────────────
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
        """Trigger anchor model init on all FSDP workers."""
        self._trainer.actor_rollout_ref_wg.init_anchor()


# =========================================================================
# Resolve ref/anchor weight-copy actions per task
# =========================================================================

def _resolve_weight_actions(
    ref_cfg: AnchorRefConfig,
    task_id: int,
) -> dict[str, bool]:
    """Anchor/ref actions for this task.

    Before ``activate_from_task`` the retention anchor is unused. From that
    task onward the actor's previous-task weights are copied into the anchor.
    The KL reference always stays the original pretrained base.
    """
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
# Main CIL loop (persistent workers, no ray restart)
# =========================================================================

def _run_cil(
    ppo_config: PPOConfig,
    known_args: argparse.Namespace,
    ref_cfg: AnchorRefConfig,
    cil_config: Optional[dict] = None,
) -> None:
    """Class-incremental learning with persistent workers."""
    import shutil

    current_time = time.strftime("%Y-%m-%d-%H%M%S")
    save_dir = known_args.save_dir
    if save_dir is None:
        raise ValueError("Error: --save_dir is required for class-incremental learning.")
    save_dir = f"{save_dir}/{current_time}"

    os.makedirs(save_dir, exist_ok=True)
    cil_info_dir = os.path.join(save_dir, "cil_info")
    metrics_dir = os.path.join(cil_info_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    base._snapshot_launch_script(cil_info_dir, explicit_script_path=known_args.launch_script)
    base._snapshot_repro_files(cil_info_dir, ppo_config)

    if cil_config is not None:
        with open(os.path.join(cil_info_dir, "cil_cfg.json"), "w", encoding="utf-8") as f:
            json.dump(cil_config, f, indent=2, ensure_ascii=False)
        print(f"[CIL] Saved CIL config to {os.path.join(cil_info_dir, 'cil_cfg.json')}")

    # ── Class order & task splits ───────────────────────────────────────
    class_names = base._list_class_names_from_dir(ppo_config.data.train_files)
    total_classes = len(class_names)
    total_tasks, base_classes, incremental_classes = base._resolve_task_plan(
        total_classes, known_args.base_classes, known_args.incremental_classes
    )
    class_order = base._build_class_order(total_classes, known_args.class_order, known_args.class_order_seed)
    class_splits = base._chunk_classes(class_order, base_classes, incremental_classes, total_tasks)
    class_order_names = [class_names[i] for i in class_order]

    with open(os.path.join(cil_info_dir, "class_order.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_order_seed": known_args.class_order_seed,
                "final_class_order_ids": class_order,
                "final_class_order_names": class_order_names,
            },
            f, indent=2,
        )

    all_labels_path = os.path.join(cil_info_dir, "all_labels.json")
    with open(all_labels_path, "w", encoding="utf-8") as f:
        json.dump(class_order_names, f, ensure_ascii=True)

    # Precompute per-task sample counts
    task_samples_info: list[dict[str, Any]] = []
    for idx, ids in enumerate(class_splits):
        ordered_class_names = [class_names[i] for i in ids]
        allowed_norm = {base._normalize_label_name(name) for name in ordered_class_names}
        train_samples = base._count_class_samples(ppo_config.data.train_files, allowed_norm)
        val_samples = base._count_class_samples(ppo_config.data.val_files, allowed_norm)
        task_samples_info.append({
            "task": idx + 1,
            "classes": ordered_class_names,
            "train_samples": train_samples,
            "val_samples": val_samples,
        })

    with open(os.path.join(cil_info_dir, "task_samples.json"), "w", encoding="utf-8") as f:
        json.dump(task_samples_info, f, indent=2)

    # ── CIL metrics tracking ───────────────────────────────────────────
    overall_last_acc_history: list[float] = []
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
    task_acc_matrix: list[list[float]] = []

    # ── Wandb ──────────────────────────────────────────────────────────
    wandb_available = False
    if os.environ.get("WANDB_DISABLED", "false").lower() not in {"true", "1", "yes"}:
        try:
            import wandb
            shared_run_id = wandb.util.generate_id()
            shared_run_name = f"{ppo_config.trainer.experiment_name}-cil-{time.strftime('%Y%m%d-%H%M%S')}"
            os.environ["WANDB_RUN_ID"] = shared_run_id
            os.environ.setdefault("WANDB_RESUME", "allow")
            os.environ.setdefault("WANDB_RUN_NAME", shared_run_name)
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "120")
            wandb_available = True
        except Exception as exc:
            print(f"Warning: failed to prepare wandb env: {exc}")

    # ── Ensure Ray is running ──────────────────────────────────────────
    base._ensure_ray()

    # ── Create persistent runner and initialise workers ONCE ───────────
    runner = PersistentRunner.remote()
    ray.get(runner.init.remote(ppo_config))

    # Initialise the frozen anchor model on all workers. A failure here is fatal:
    # the retention reward requires anchor log-probs from task 2 onward.
    ray.get(runner.init_anchor_on_workers.remote())

    print(f"[CIL] Starting {total_tasks} CIL tasks with persistent workers.")

    # ── Task loop ──────────────────────────────────────────────────────
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

        # Build evaluation plan (identical to original)
        seen_class_names = [class_names[i] for i in seen_class_ids]
        eval_plan = []
        for eval_idx, eval_class_ids in enumerate(class_splits[: task_idx + 1]):
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

        # Determine weight copy actions
        weight_actions = _resolve_weight_actions(ref_cfg, task_id)

        # Run the task
        result = ray.get(
            runner.run_task.remote(
                config=task_config,
                task_id=task_id,
                val_progress=not known_args.no_val_progress,
                val_write_predictions=known_args.val_write_predictions,
                val_output_dir=known_args.val_output_dir,
                allowed_classes=ordered_task_class_names,
                allowed_class_order=prompt_label_list or ordered_task_class_names,
                extra_val_splits=eval_plan,
                prompt_label_list=prompt_label_list,
                copy_actor_to_anchor=weight_actions["copy_actor_to_anchor"],
                enable_anchor=weight_actions["enable_anchor"],
            )
        )

        extra_eval_results = result.get("extra_eval_results", {}) if isinstance(result, dict) else {}
        last_checkpoint = result.get("last_checkpoint_path") if isinstance(result, dict) else None

        # ── CIL metrics (identical to original) ────────────────────────
        single_acc = extra_eval_results.get(f"task{task_id}", {}).get(
            "val/cls_accuracy",
            extra_eval_results.get(f"task{task_id}", {}).get("val/accuracy_reward", 0.0),
        )
        overall_last_acc = extra_eval_results.get("overall_seen", {}).get(
            "val/cls_accuracy",
            extra_eval_results.get("overall_seen", {}).get("val/accuracy_reward", 0.0),
        )
        overall_last_acc_history.append(overall_last_acc)
        cumulative_avg = sum(overall_last_acc_history) / len(overall_last_acc_history)

        per_task_acc_raw = {
            f"task{j}": extra_eval_results.get(f"task{j}", {}).get(
                "val/cls_accuracy",
                extra_eval_results.get(f"task{j}", {}).get("val/accuracy_reward", 0.0),
            )
            for j in range(1, task_id + 1)
        }
        per_task_acc_pct = {k: round(v * 100, 4) for k, v in per_task_acc_raw.items()}

        current_row = [per_task_acc_raw[f"task{j}"] for j in range(1, task_id + 1)]
        task_acc_matrix.append(current_row)

        fr: Optional[float] = None
        if task_id >= 2:
            np_acctable = np.zeros([task_id, task_id], dtype=np.float32)
            for idxx, line in enumerate(task_acc_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line, dtype=np.float32)
            np_acctable = np_acctable.T
            fr = float(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task_id - 1])[: task_id - 1]))

        def _pct(v: Optional[float]) -> Optional[float]:
            return round(v * 100, 4) if v is not None else None

        summary["tasks"][f"task{task_id}"] = {
            "last_acc": _pct(overall_last_acc),
            "average_acc": _pct(cumulative_avg),
            "last_acc_details": per_task_acc_pct,
            "forgetting_rate": _pct(fr),
        }

        current_samples = task_samples_info[task_idx]
        train_samples = current_samples["train_samples"]
        val_samples = current_samples["val_samples"]

        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        task_log_path = os.path.join(metrics_dir, "task_metrics.log")
        with open(task_log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "task": task_id,
                    "last_acc": _pct(overall_last_acc),
                    "average_acc": _pct(cumulative_avg),
                    "last_acc_details": per_task_acc_pct,
                    "forgetting_rate": _pct(fr),
                })
                + "\n"
            )

        decomp_parts = [f"T{j}:{per_task_acc_pct[f'task{j}']:.2f}%" for j in range(1, task_id + 1)]
        extra_metrics_str = ""
        if fr is not None:
            extra_metrics_str += f" | FR: {_pct(fr):.2f}%"
        log_msg = (
            f"[Task {task_id} Finished] "
            + f"Last acc: {_pct(overall_last_acc):.2f}% | Average acc: {_pct(cumulative_avg):.2f}%"
            + f" | Decomp: {' '.join(decomp_parts)}"
            + extra_metrics_str
        )
        print(log_msg)

        # ── Checkpoint pruning (identical to original) ──────────────────
        if known_args.save_task_ckpt in ("latest", "none") and last_checkpoint:
            try:
                ckpt_dirname = os.path.basename(last_checkpoint.rstrip(os.sep))
                for entry in os.listdir(task_save_dir):
                    full_path = os.path.join(task_save_dir, entry)
                    if not os.path.isdir(full_path):
                        continue
                    if entry.startswith("global_step_") and entry != ckpt_dirname:
                        shutil.rmtree(full_path)
            except Exception as exc:
                print(f"Warning: failed to prune old checkpoints in {task_save_dir}: {exc}")

        if known_args.save_task_ckpt in ("latest", "none") and previous_task_dir and previous_task_dir != task_save_dir:
            try:
                for entry in os.listdir(previous_task_dir):
                    full_path = os.path.join(previous_task_dir, entry)
                    if os.path.isdir(full_path) and entry.startswith("global_step_"):
                        shutil.rmtree(full_path)
            except Exception as exc:
                print(f"Warning: failed to prune checkpoints in previous task dir {previous_task_dir}: {exc}")

        previous_task_dir = task_save_dir

        if wandb_available:
            try:
                import wandb
                if wandb.run is not None:
                    wandb_payload: dict[str, Any] = {
                        "cil/task": task_id,
                        "cil/single_acc": _pct(single_acc),
                        "cil/overall_last_acc": _pct(overall_last_acc),
                        "cil/cumulative_average_acc": _pct(cumulative_avg),
                        "cil/train_samples": train_samples,
                        "cil/val_samples": val_samples,
                    }
                    if fr is not None:
                        wandb_payload["cil/forgetting_rate"] = _pct(fr)
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
            print(f"Warning: failed to prune final task checkpoints in {previous_task_dir}: {exc}")

    # ── Clean up ───────────────────────────────────────────────────────
    ray.kill(runner)
    print(f"[CIL] All {total_tasks} tasks completed.")


# =========================================================================
# Argument parsing (--cil_cfg provides defaults; CLI args override)
# =========================================================================

def _parse_args() -> "tuple[argparse.Namespace, list[str]]":
    # ── Phase 1: extract --cil_cfg path (before full parse) ───────────────
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--cil_cfg", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()

    cfg_defaults: dict[str, Any] = {}
    if pre_args.cil_cfg is not None:
        cfg_defaults = _load_cil_cfg(pre_args.cil_cfg)
        print(f"[CIL-CFG] Loaded defaults from {pre_args.cil_cfg}")

    # ── Phase 2: full parser (cfg values become defaults, CLI overrides) ──
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cil_cfg", type=str, default=None,
        help="Path to cil_cfg.json. Values serve as defaults; CLI args override.",
    )

    # ── standard CIL args ─────────────────────────────────────────────
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

    # ── CTAN args ──────────────────────────────────────────────────────
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

    # ── Retention reward args ───────────────────────────────────────────
    parser.add_argument("--retention_reward_enable", action="store_true", default=False)
    parser.add_argument("--retention_lambda", type=float, default=0.5)
    parser.add_argument("--retention_alpha", type=float, default=20.0)
    parser.add_argument("--retention_activate_from_task", type=int, default=2)

    # ── ref / anchor activation ────────────────────────────────────────
    # ref = frozen original pretrained; anchor = frozen previous-task checkpoint.
    parser.add_argument("--rapo_activate_from_task", type=int, default=2)

    # ── Apply cil_cfg defaults (CLI args still take precedence) ───────────
    if cfg_defaults:
        parser.set_defaults(**cfg_defaults)

    return parser.parse_known_args()


def _set_env(known_args: argparse.Namespace) -> None:
    """Mirror env vars from CLI for per-worker config reading."""
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

    original_model_path = ppo_config.worker.actor.model.model_path

    print(
        "[CIL] Configuration:\n"
        f"  reference=original pretrained base (frozen KL anchor)\n"
        f"  anchor=previous-task policy (frozen retention target)\n"
        f"  activate_from_task={ref_cfg.activate_from_task}\n"
        f"  original_model_path={original_model_path}"
    )

    # Collect all CIL hyperparameters for reproducibility
    cil_config = _build_cil_cfg(known_args)

    _run_cil(ppo_config, known_args, ref_cfg, cil_config=cil_config)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

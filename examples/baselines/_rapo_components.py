"""_rapo_components.py
===============================================================================
Shared components for the RaPO training path (class-incremental learning).

Default RaPO behaviour:

    * standard GRPO with loss-level KL to a frozen reference
    * the KL reference is always the original pretrained base model
    * the retention anchor is always the previous-task checkpoint, active from
      ``rapo_activate_from_task`` (default 2) onward
    * CTAN advantage normalisation (EMA of cross-task reward statistics) is
      enabled when ``--ctan_enable`` is passed. The compute path is fixed:
      post-group-norm, scale by EMA std only, seed from the previous task's
      last batch (task ≥ 2). These three choices are not config knobs.

Architecture note
-----------------
VERL initialises both actor and ref from ``worker.actor.model.model_path``.
After init, ``load_checkpoint`` restores ONLY the actor. Redirecting
``model_path`` -> anchor/ref checkpoint therefore only affects the frozen
ref/anchor model, while the actor weights come from the task checkpoint.

This module provides CTAN (``EMAAdvConfig`` / ``EMAAdvNormalizer``), the
retention reward (``RetentionRewardConfig`` / ``_apply_retention_reward``), the
GRPO trainer (``RayPPOContinualRaPOTrainer``), the cil_cfg loader, and EMA-state
persistence. The persistent-worker CIL entry points live
in ``img_cls_cil/``, ``cil_det/`` and ``video_cls_cil/``.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

import examples.baselines.img_cls_cil.image_cls_cil as base
import verl.trainer.ray_trainer as ray_trainer_module
from verl.protocol import DataProto
from verl.trainer.core_algos import AdvantageEstimator


# ---------------------------------------------------------------------------
# CIL config save / load helpers
# ---------------------------------------------------------------------------

# Mapping from grouped JSON keys → argparse dest names.
_CIL_CFG_GROUP_MAP: dict[str, dict[str, str]] = {
    "cil": {
        "class_order_seed": "class_order_seed",
        "class_order": "class_order",
        "save_task_ckpt": "save_task_ckpt",
        "base_classes": "base_classes",
        "incremental_classes": "incremental_classes",
        "prompt_seen_labels": "prompt_seen_labels",
    },
    "ctan": {
        "enable": "ema_adv_enable",
        "beta": "ema_adv_beta",
        "eps": "ema_adv_eps",
        "bootstrap_steps": "ema_adv_bootstrap_steps",
        "activate_from_task": "ema_adv_activate_from_task",
        "min_std": "ema_adv_min_std",
        "guard_abs_max": "ema_adv_guard_abs_max",
        "bias_correction": "ema_adv_bias_correction",
        "beta_warmup_steps": "ema_adv_beta_warmup_steps",
        "beta_warmup_init": "ema_adv_beta_warmup_init",
    },
    "retention_reward": {
        "enable": "retention_reward_enable",
        "lambda_coef": "retention_lambda",
        "kl_scale": "retention_alpha",
        "activate_from_task": "retention_activate_from_task",
    },
    "anchor_ref": {
        "activate_from_task": "rapo_activate_from_task",
    },
}

# Legacy CTAN keys. The compute path is fixed; leftover values in a cfg
# file are ignored and must not become argparse dests.
_CIL_CFG_HARDCODED_CTAN_KEYS = frozenset({"position", "norm_target", "init_strategy"})


def _load_cil_cfg(path: str) -> dict[str, Any]:
    """Load ``cil_cfg.json`` and flatten grouped keys to argparse dest names.

    Supports both *grouped* format (recommended)::

        {"ctan": {"beta": 0.999, ...}, "retention_reward": {...}, ...}

    and *flat* format (keys are argparse dest names directly)::

        {"ema_adv_beta": 0.999, "retention_lambda": 1.0, ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        raw: dict = json.load(f)
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and key in _CIL_CFG_GROUP_MAP:
            mapping = _CIL_CFG_GROUP_MAP[key]
            for k, v in value.items():
                if k in mapping:
                    flat[mapping[k]] = v
                elif key == "ctan" and k in _CIL_CFG_HARDCODED_CTAN_KEYS:
                    print(
                        f"[CIL-CFG] Ignoring '{key}.{k}' in {path}: "
                        "CTAN position/norm_target/init_strategy are fixed in code"
                    )
                else:
                    print(f"[CIL-CFG] Warning: unknown key '{key}.{k}' in {path}")
        else:
            # Flat leftover legacy keys must not become argparse dests.
            if key in _CIL_CFG_HARDCODED_CTAN_KEYS or key in {
                "ema_adv_position",
                "ema_adv_norm_target",
                "ema_adv_init_strategy",
            }:
                print(
                    f"[CIL-CFG] Ignoring '{key}' in {path}: "
                    "CTAN position/norm_target/init_strategy are fixed in code"
                )
                continue
            # Flat key – pass through directly as argparse dest
            flat[key] = value
    return flat


def _build_cil_cfg(known_args: argparse.Namespace) -> dict[str, Any]:
    """Collect all CIL hyperparameters from parsed args into a grouped dict."""
    return {
        "cil": {
            "class_order_seed": known_args.class_order_seed,
            "class_order": known_args.class_order,
            "save_task_ckpt": known_args.save_task_ckpt,
            "base_classes": known_args.base_classes,
            "incremental_classes": known_args.incremental_classes,
            "prompt_seen_labels": known_args.prompt_seen_labels,
        },
        "ctan": {
            "enable": known_args.ema_adv_enable,
            "beta": known_args.ema_adv_beta,
            "eps": known_args.ema_adv_eps,
            "bootstrap_steps": known_args.ema_adv_bootstrap_steps,
            "activate_from_task": known_args.ema_adv_activate_from_task,
            "min_std": known_args.ema_adv_min_std,
            "guard_abs_max": known_args.ema_adv_guard_abs_max,
            "bias_correction": known_args.ema_adv_bias_correction,
            "beta_warmup_steps": known_args.ema_adv_beta_warmup_steps,
            "beta_warmup_init": known_args.ema_adv_beta_warmup_init,
        },
        "retention_reward": {
            "enable": known_args.retention_reward_enable,
            "lambda_coef": known_args.retention_lambda,
            "kl_scale": known_args.retention_alpha,
            "activate_from_task": known_args.retention_activate_from_task,
        },
        "anchor_ref": {
            "activate_from_task": known_args.rapo_activate_from_task,
        },
    }


# ---------------------------------------------------------------------------
# Reference / anchor policy configuration
# ---------------------------------------------------------------------------

@dataclass
class AnchorRefConfig:
    """RaPO frozen-policy configuration.

    RaPO keeps two frozen policies alongside the trainable actor:

    - reference: the original pretrained base model, used as the PPO KL anchor.
    - anchor:    the immediately preceding task's policy, used as the retention
                 target (drift is measured against it).

    Both roles are fixed by design; only the first task at which they activate
    is configurable.
    """

    activate_from_task: int = 2


# ---------------------------------------------------------------------------
# EMA advantage normaliser
# ---------------------------------------------------------------------------

@dataclass
class EMAAdvConfig:
    """CTAN hyperparameters.

    Position, norm target, and init strategy are not fields: the normaliser
    always does post-group-norm, scales by EMA std only, and seeds from the
    previous task's last batch on task ≥ 2.
    """

    enabled: bool = False
    beta: float = 0.999
    eps: float = 1e-6
    bootstrap_steps: int = 0
    activate_from_task: int = 2
    min_std: float = 1e-3
    guard_abs_max: float = 5.0
    bias_correction: bool = True
    beta_warmup_steps: int = 2
    beta_warmup_init: float = 0.9

    @staticmethod
    def from_env() -> "EMAAdvConfig":
        enabled = os.environ.get("EMA_ADV_ENABLED", "0").lower() in {"1", "true", "yes"}
        beta = float(os.environ.get("EMA_ADV_BETA", "0.999"))
        eps = float(os.environ.get("EMA_ADV_EPS", "1e-6"))
        bootstrap_steps = int(os.environ.get("EMA_ADV_BOOTSTRAP_STEPS", "0"))
        activate_from_task = int(os.environ.get("EMA_ADV_ACTIVATE_FROM_TASK", "2"))
        min_std = float(os.environ.get("EMA_ADV_MIN_STD", "1e-3"))
        guard_abs_max = float(os.environ.get("EMA_ADV_GUARD_ABS_MAX", "5.0"))
        bias_correction = os.environ.get("EMA_ADV_BIAS_CORRECTION", "1").lower() in {"1", "true", "yes"}
        beta_warmup_steps = int(os.environ.get("EMA_ADV_BETA_WARMUP_STEPS", "2"))
        beta_warmup_init = float(os.environ.get("EMA_ADV_BETA_WARMUP_INIT", "0.9"))
        return EMAAdvConfig(
            enabled=enabled,
            beta=beta,
            eps=eps,
            bootstrap_steps=bootstrap_steps,
            activate_from_task=activate_from_task,
            min_std=min_std,
            guard_abs_max=guard_abs_max,
            bias_correction=bias_correction,
            beta_warmup_steps=beta_warmup_steps,
            beta_warmup_init=beta_warmup_init,
        )


class EMAAdvNormalizer:
    def __init__(
        self,
        config: EMAAdvConfig,
        initial_state: Optional[dict[str, Any]] = None,
        task_id: Optional[int] = None,
    ):
        self.config = config
        self.task_id = task_id
        self.ema_mean = 0.0
        self.ema_std = 1.0
        self.update_count = 0
        self._initialized = False
        self._last_batch_reward_mean: Optional[float] = None
        self._last_batch_reward_std: Optional[float] = None
        self._task_start_update_count: int = 0
        self._fresh_start: bool = True
        if initial_state:
            self.ema_mean = float(initial_state.get("ema_mean", self.ema_mean))
            self.ema_std = float(initial_state.get("ema_std", self.ema_std))
            self.update_count = int(initial_state.get("update_count", self.update_count))
            self._initialized = True
            last_mean = initial_state.get("last_batch_reward_mean")
            last_std = initial_state.get("last_batch_reward_std")
            if last_mean is not None:
                self._last_batch_reward_mean = float(last_mean)
            if last_std is not None:
                self._last_batch_reward_std = float(last_std)
            self._task_start_update_count = self.update_count
            self._fresh_start = False
            # task1_last_batch: task ≥ 2 reseeds from the previous task's last batch.
            if (
                self.task_id is not None
                and self.task_id >= 2
                and self._last_batch_reward_mean is not None
                and self._last_batch_reward_std is not None
            ):
                self.ema_mean = float(self._last_batch_reward_mean)
                self.ema_std = float(self._last_batch_reward_std)
                self._initialized = True

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema_mean": float(self.ema_mean),
            "ema_std": float(self.ema_std),
            "update_count": int(self.update_count),
            "beta": float(self.config.beta),
            "eps": float(self.config.eps),
            "bootstrap_steps": int(self.config.bootstrap_steps),
            "activate_from_task": int(self.config.activate_from_task),
            "min_std": float(self.config.min_std),
            "guard_abs_max": float(self.config.guard_abs_max),
            "bias_correction": self.config.bias_correction,
            "beta_warmup_steps": int(self.config.beta_warmup_steps),
            "beta_warmup_init": float(self.config.beta_warmup_init),
            "last_batch_reward_mean": self._last_batch_reward_mean,
            "last_batch_reward_std": self._last_batch_reward_std,
        }

    def should_use_ema_grpo(self) -> bool:
        if self.task_id is None:
            return True
        return self.task_id >= int(self.config.activate_from_task)

    def observe_batch_without_ema(self, data: DataProto) -> None:
        scores = data.batch["token_level_rewards"].sum(dim=-1)
        self._last_batch_reward_mean = float(scores.mean().item())
        self._last_batch_reward_std = float(scores.std().item())

    def _apply_init_strategy(self, current_mean: float, current_std: float) -> None:
        # task1_last_batch: seed EMA from the previous task's final batch stats
        # when available, otherwise from the current batch.
        if self._initialized:
            return
        if self._last_batch_reward_mean is not None and self._last_batch_reward_std is not None:
            self.ema_mean = float(self._last_batch_reward_mean)
            self.ema_std = float(self._last_batch_reward_std)
        else:
            self.ema_mean = float(current_mean)
            self.ema_std = float(current_std)
        self._initialized = True

    def _effective_beta(self) -> float:
        # Linearly warm beta from beta_warmup_init up to beta over the first
        # beta_warmup_steps updates of each task.
        if self.config.beta_warmup_steps <= 0:
            return self.config.beta
        task_step = self.update_count - self._task_start_update_count
        if task_step >= self.config.beta_warmup_steps:
            return self.config.beta
        frac = task_step / self.config.beta_warmup_steps
        return self.config.beta_warmup_init + (self.config.beta - self.config.beta_warmup_init) * frac

    def _bias_corrected_stats(self) -> tuple[float, float]:
        if not self.config.bias_correction or not self._fresh_start:
            return self.ema_mean, self.ema_std
        t = self.update_count - self._task_start_update_count
        if t == 0:
            return self.ema_mean, self.ema_std
        beta = self.config.beta
        bt = beta ** t
        denom = 1.0 - bt
        if denom < 1e-12:
            return self.ema_mean, self.ema_std
        corrected_mean = self.ema_mean / denom
        corrected_std = (self.ema_std - bt * 1.0) / denom
        corrected_std = max(corrected_std, float(self.config.min_std))
        return corrected_mean, corrected_std

    def _update_ema(self, current_mean: float, current_std: float) -> None:
        # norm_target=std: update ema_std only (ema_mean is unused for scaling).
        self._apply_init_strategy(current_mean, current_std)
        beta = self._effective_beta()
        self.ema_std = beta * self.ema_std + (1.0 - beta) * float(current_std)
        self.ema_std = max(self.ema_std, float(self.config.min_std))

    def _group_mean_std(self, scores: torch.Tensor, index: Any) -> tuple[torch.Tensor, torch.Tensor]:
        group_map: dict[Any, list[int]] = {}
        for i in range(scores.shape[0]):
            group_map.setdefault(index[i], []).append(i)
        group_mean = torch.zeros_like(scores)
        group_std = torch.zeros_like(scores)
        for _, row_ids in group_map.items():
            g = scores[row_ids]
            assert g.numel() > 1, "GRPO needs rollout.n > 1."
            group_mean[row_ids] = torch.mean(g)
            group_std[row_ids] = torch.std(g)
        return group_mean, group_std

    def _apply_guard_rail(
        self,
        scaled_adv: torch.Tensor,
        centered: torch.Tensor,
        group_std: torch.Tensor,
        index: Any,
    ) -> torch.Tensor:
        if self.config.guard_abs_max <= 0:
            return scaled_adv
        group_map: dict[Any, list[int]] = {}
        for i in range(scaled_adv.shape[0]):
            group_map.setdefault(index[i], []).append(i)
        final_adv = scaled_adv.clone()
        n_fallback = 0
        for _, row_ids in group_map.items():
            if torch.any(torch.abs(scaled_adv[row_ids]) > self.config.guard_abs_max):
                g_std = group_std[row_ids[0]]
                final_adv[row_ids] = centered[row_ids] / (g_std + self.config.eps)
                n_fallback += 1
        if n_fallback > 0:
            print(f"[EMA-ADV] Guard rail triggered for {n_fallback}/{len(group_map)} groups.")
        return final_adv

    def _is_bootstrap_done(self) -> bool:
        return self.update_count >= self.config.bootstrap_steps

    def compute_grpo_advantage(self, data: DataProto) -> DataProto:
        token_level_rewards = data.batch["token_level_rewards"]
        response_mask = data.batch["response_mask"]
        index = data.non_tensor_batch["uid"]
        scores = token_level_rewards.sum(dim=-1)
        self._last_batch_reward_mean = float(scores.mean().item())
        self._last_batch_reward_std = float(scores.std().item())
        use_ema_now = self._is_bootstrap_done()

        # post_group_norm: center by group mean, then scale by the EMA std.
        group_mean, group_std = self._group_mean_std(scores, index)
        centered = scores - group_mean
        self._update_ema(float(scores.mean().item()), float(scores.std().item()))
        if use_ema_now:
            _, corrected_std = self._bias_corrected_stats()
            std_value = max(float(corrected_std), float(self.config.min_std))
            scaled = centered / (std_value + self.config.eps)
            final_adv = self._apply_guard_rail(scaled, centered, group_std, index)
        else:
            final_adv = centered / (group_std + self.config.eps)

        self.update_count += 1
        token_adv = final_adv.unsqueeze(-1) * response_mask
        data.batch["advantages"] = token_adv
        data.batch["returns"] = token_adv
        return data


# ---------------------------------------------------------------------------
# Retention reward
# ---------------------------------------------------------------------------

def _first_defined_env(default: str, *names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value
    return default

@dataclass
class RetentionRewardConfig:
    """Configuration for the per-step retention reward.

    R_retention = exp(-kl_scale * D_drift), with D_drift the one-sided
    per-sequence drift of the current actor from the frozen anchor policy.
    The total reward is additive: R_task + lambda_coef * R_retention.
    """
    enabled: bool = False
    lambda_coef: float = 0.5
    kl_scale: float = 20.0
    activate_from_task: int = 2

    @staticmethod
    def from_env() -> "RetentionRewardConfig":
        enabled = _first_defined_env("0", "RETENTION_REWARD_ENABLED").lower() in {"1", "true", "yes"}
        lambda_coef = float(_first_defined_env("0.5", "RETENTION_REWARD_LAMBDA"))
        kl_scale = float(_first_defined_env("20.0", "RETENTION_ALPHA"))
        activate_from_task = int(
            _first_defined_env("2", "RETENTION_REWARD_ACTIVATE_FROM_TASK")
        )
        return RetentionRewardConfig(
            enabled=enabled,
            lambda_coef=lambda_coef,
            kl_scale=kl_scale,
            activate_from_task=activate_from_task,
        )


def _apply_retention_reward(
    data: DataProto,
    cfg: RetentionRewardConfig,
) -> "tuple[DataProto, dict[str, float]]":
    """Add the retention reward to token_level_rewards.

    Drift is actor_lp - anchor_lp, clamped at zero so only movement away from
    the anchor is penalised; R_retention = exp(-kl_scale * drift) is high when
    the actor stays close to the anchor. Log-probs are detached because the
    retention term is a scalar reward, not a differentiable loss.
    """
    if "anchor_log_probs" not in data.batch:
        raise RuntimeError(
            "Retention reward is active but 'anchor_log_probs' is missing from the "
            "batch. The frozen anchor (previous-task) log-probs must be computed "
            "before advantage; check init_anchor() and the per-task actor->anchor "
            "weight copy."
        )
    old_lp = data.batch["old_log_probs"].detach()          # (B, T)
    anchor_lp = data.batch["anchor_log_probs"].detach()    # (B, T)
    mask = data.batch["response_mask"]                     # (B, T)

    drift_per_token = (old_lp - anchor_lp) * mask  # positive when actor diverges
    num_resp = mask.sum(dim=-1).clamp(min=1)
    drift_per_seq = (drift_per_token.sum(dim=-1) / num_resp).clamp(min=0.0)

    r_retention = torch.exp(-cfg.kl_scale * drift_per_seq)  # (B,) in (0, 1]

    token_retention = (
        cfg.lambda_coef * r_retention.unsqueeze(-1) / num_resp.unsqueeze(-1)
    ) * mask
    data.batch["token_level_rewards"] = data.batch["token_level_rewards"] + token_retention

    metrics = {
        "reward/retention_drift": drift_per_seq.mean().item(),
        "reward/retention_reward": r_retention.mean().item(),
    }
    return data, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RayPPOContinualRaPOTrainer(base.RayPPOContinualTrainer):
    """GRPO trainer with optional EMA-ADV and retention reward."""

    def __init__(
        self,
        *args,
        ema_adv_config: Optional[EMAAdvConfig] = None,
        ema_adv_state: Optional[dict[str, Any]] = None,
        retention_cfg: Optional[RetentionRewardConfig] = None,
        **kwargs,
    ):
        task_id = kwargs.pop("task_id", None)
        super().__init__(*args, **kwargs)
        self.ema_adv_config = ema_adv_config or EMAAdvConfig(enabled=False)
        self.ema_adv = (
            EMAAdvNormalizer(self.ema_adv_config, initial_state=ema_adv_state, task_id=task_id)
            if self.ema_adv_config.enabled
            else None
        )
        self.retention_cfg = retention_cfg
        self._task_id: Optional[int] = task_id
        self._orig_compute_advantage = None
        self._pending_retention_metrics: dict = {}
        try:
            t_path = self.config.worker.actor.model.anchor_model_path
            r_path = self.config.worker.actor.model.model_path
            self._anchor_available: bool = bool(t_path and str(t_path).strip() and t_path != r_path)
        except AttributeError:
            self._anchor_available = False

    def _compute_advantage_with_hooks(
        self,
        data: DataProto,
        adv_estimator: AdvantageEstimator,
        gamma: float = 1.0,
        lam: float = 1.0,
    ) -> DataProto:
        # Lazy-patch logger.log so retention metrics are included in WandB/tensorboard
        if (
            self.retention_cfg is not None
            and self.retention_cfg.enabled
            and not hasattr(self, "_orig_logger_log")
            and hasattr(self, "logger")
        ):
            _self = self
            _orig = self.logger.log

            def _patched_log(data, step):
                if _self._pending_retention_metrics:
                    data = {**data, **_self._pending_retention_metrics}
                    _self._pending_retention_metrics.clear()
                _orig(data=data, step=step)

            self._orig_logger_log = _orig
            self.logger.log = _patched_log

        # Fetch anchor log probs when anchor != ref (separate frozen model).
        if (
            self._anchor_available
            and self.retention_cfg is not None
            and self.retention_cfg.enabled
            and "anchor_log_probs" not in data.batch
            and "old_log_probs" in data.batch
            and (self._task_id is None or self._task_id >= self.retention_cfg.activate_from_task)
        ):
            anchor_out = self.actor_rollout_ref_wg.compute_anchor_log_probs(data)
            data = data.union(anchor_out)

        # Inject retention reward before advantage computation
        if (
            self.retention_cfg is not None
            and self.retention_cfg.enabled
            and self.retention_cfg.lambda_coef > 0
            and "ref_log_probs" in data.batch
            and "old_log_probs" in data.batch
            and (self._task_id is None or self._task_id >= self.retention_cfg.activate_from_task)
        ):
            data, dm = _apply_retention_reward(data, self.retention_cfg)
            self._pending_retention_metrics.update(dm)

        # EMA advantage normalisation
        if adv_estimator == AdvantageEstimator.GRPO and self.ema_adv is not None:
            if self.ema_adv.should_use_ema_grpo():
                return self.ema_adv.compute_grpo_advantage(data)
            self.ema_adv.observe_batch_without_ema(data)
            return self._orig_compute_advantage(data, adv_estimator=adv_estimator, gamma=gamma, lam=lam)
        return self._orig_compute_advantage(data, adv_estimator=adv_estimator, gamma=gamma, lam=lam)

    def fit(self):
        self._orig_compute_advantage = ray_trainer_module.compute_advantage
        ray_trainer_module.compute_advantage = self._compute_advantage_with_hooks
        try:
            super().fit()
        finally:
            ray_trainer_module.compute_advantage = self._orig_compute_advantage
            if hasattr(self, "_orig_logger_log"):
                self.logger.log = self._orig_logger_log
                del self._orig_logger_log


# ---------------------------------------------------------------------------
# EMA state persistence helpers
# ---------------------------------------------------------------------------

def _ema_stats_file_from_task_dir(task_save_dir: str) -> str:
    task_save_dir = os.path.abspath(task_save_dir)
    parent_dir = os.path.dirname(task_save_dir)
    return os.path.join(parent_dir, "ema_online_stats.json")


def _load_ema_state(path: str) -> Optional[dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "ema_mean" in payload and "ema_std" in payload:
            return payload
        if isinstance(payload, dict):
            latest = payload.get("latest")
            if isinstance(latest, dict) and "ema_mean" in latest:
                return latest
            task_history = payload.get("task_history", [])
            if isinstance(task_history, list) and task_history:
                last = task_history[-1]
                if isinstance(last, dict) and "ema_mean" in last:
                    return last
        return None
    except Exception as exc:
        print(f"[EMA-ADV] Warning: failed to load EMA state from {path}: {exc}")
        return None


def _save_ema_state(path: str, payload: dict[str, Any], task_id: Optional[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_payload: dict[str, Any] = {"latest": payload, "task_history": []}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                if "ema_mean" in existing and "ema_std" in existing:
                    file_payload = {"latest": payload, "task_history": [existing]}
                else:
                    history = existing.get("task_history", [])
                    file_payload = {"latest": payload, "task_history": history if isinstance(history, list) else []}
        except Exception as exc:
            print(f"[EMA-ADV] Warning: failed to read existing history from {path}: {exc}")

    entry = dict(payload)
    entry["task"] = task_id
    history = file_payload.get("task_history", [])
    if not isinstance(history, list):
        history = []
    if task_id is not None:
        replaced = False
        for idx, item in enumerate(history):
            if isinstance(item, dict) and item.get("task") == task_id:
                history[idx] = entry
                replaced = True
                break
        if not replaced:
            history.append(entry)
    else:
        history.append(entry)
    file_payload["task_history"] = history
    file_payload["latest"] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(file_payload, f, indent=2)

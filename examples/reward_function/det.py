"""Detection reward function for CIL object detection.

compute_score combines three terms: an IoU reward and a category-accuracy
reward, both computed by Hungarian matching predictions to ground truth within
the same category and normalising by max(num_gt, num_pred), plus a format
reward for well-formed <think>/<answer> output.
"""

import json
import re
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

REWARD_NAME = "det"
REWARD_TYPE = "sequential"

LAZY_THINK_PATTERNS = [
    "step-by-step reasoning here",
    "step by step reasoning here",
    "reasoning here",
    "thinking process here",
    "detailed reasoning here",
]


def _extract_tag_content(response: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_detections(text: str) -> list[dict] | None:
    """Parse detection JSON from model output. Returns None on failure."""
    text = text.strip()
    if not text:
        return []
    # Fix common formatting issues
    text = text.replace("'", '"')
    text = re.sub(r",\s*]", "]", text)  # trailing comma
    text = re.sub(r",\s*}", "}", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, list):
        return None
    valid = []
    for item in result:
        if not isinstance(item, dict):
            continue
        cat = item.get("category")
        bbox = item.get("bbox")
        if cat is None or bbox is None:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            bbox = [max(0, min(1000, int(round(float(v))))) for v in bbox]
        except (ValueError, TypeError):
            continue
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        valid.append({"category": str(cat).strip().lower(), "bbox": bbox})
    return valid


def _compute_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _hungarian_match(gt_dets: list[dict], pred_dets: list[dict]) -> tuple[float, float]:
    """Match predictions to ground truth using Hungarian algorithm.

    Only matches within the same category. Returns (iou_reward, cls_reward).
    """
    n_gt = len(gt_dets)
    n_pred = len(pred_dets)
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0
    if n_gt == 0:
        return 0.0, 0.0
    if n_pred == 0:
        return 0.0, 0.0

    # Build cost matrix (negative IoU for same-category pairs, 0 for cross-category)
    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, gt in enumerate(gt_dets):
        for j, pred in enumerate(pred_dets):
            iou = _compute_iou(gt["bbox"], pred["bbox"])
            iou_matrix[i, j] = iou
            if gt["category"] == pred["category"]:
                cost_matrix[i, j] = -iou  # negative for minimization
            # else: 0 cost (no match possible)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    total_iou = 0.0
    correct_cls = 0
    for r, c in zip(row_ind, col_ind):
        if gt_dets[r]["category"] == pred_dets[c]["category"]:
            total_iou += iou_matrix[r, c]
            if iou_matrix[r, c] >= 0.5:
                correct_cls += 1

    denom = max(n_gt, n_pred)
    iou_reward = total_iou / denom
    cls_reward = correct_cls / denom
    return min(iou_reward, 1.0), min(cls_reward, 1.0)


def format_reward(response: str) -> float:
    """Check format: <think>...</think><answer>JSON</answer>."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(pattern, response):
        return 0.0

    think_text = _extract_tag_content(response, "think")
    normalized_think = " ".join(think_text.lower().split())
    if not normalized_think:
        return 0.0
    if any(p in normalized_think for p in LAZY_THINK_PATTERNS):
        return 0.0

    answer_text = _extract_tag_content(response, "answer")
    dets = _parse_detections(answer_text)
    if dets is None:
        return 0.5  # format correct but JSON invalid
    return 1.0


def iou_reward(response: str, ground_truth: str) -> float:
    """IoU-based detection reward using Hungarian matching."""
    answer_text = _extract_tag_content(response, "answer")
    pred_dets = _parse_detections(answer_text)
    gt_dets = _parse_detections(ground_truth)
    if pred_dets is None:
        pred_dets = []
    if gt_dets is None:
        gt_dets = []
    iou_r, _ = _hungarian_match(gt_dets, pred_dets)
    return iou_r


def cls_reward(response: str, ground_truth: str) -> float:
    """Category-level detection accuracy reward."""
    answer_text = _extract_tag_content(response, "answer")
    pred_dets = _parse_detections(answer_text)
    gt_dets = _parse_detections(ground_truth)
    if pred_dets is None:
        pred_dets = []
    if gt_dets is None:
        gt_dets = []
    _, cls_r = _hungarian_match(gt_dets, pred_dets)
    return cls_r


def compute_score(
    reward_input: dict[str, Any],
    iou_weight: float = 1,
    cls_weight: float = 1,
    format_weight: float = 1,
) -> dict[str, float]:
    response = reward_input["response"]
    ground_truth = reward_input["ground_truth"]

    fmt = format_reward(response)
    iou_r = iou_reward(response, ground_truth)
    cls_r = cls_reward(response, ground_truth)

    overall = iou_weight * iou_r + cls_weight * cls_r + format_weight * fmt
    return {
        "overall": overall,
        "format": fmt,
        "iou": iou_r,
        "cls_accuracy": cls_r,
    }
"""image_det_cil.py
===============================================================================
Shared helpers and evaluation for object-detection CIL.

The main launch path is ``image_det_cil_rapo.py``. This module provides
JSONL loading, bbox mAP, and the continual eval trainer mixin. Optional SAM
mask diagnostics remain off unless a checkpoint is supplied.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.config import PPOConfig
from verl.trainer.metrics import reduce_metrics
from verl.trainer.ray_trainer import RayPPOTrainer
from verl.utils.dataset import RLHFDataset, collate_fn


# =========================================================================
# Utility helpers (shared with det CIL)
# =========================================================================

def _normalize_label_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").replace(".", " ").lower().strip()


def _extract_answer_text(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", str(text), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(text).strip()


def _parse_detections(text: str) -> list[dict]:
    """Parse detection JSON.  Returns empty list on failure."""
    text = text.strip()
    if not text:
        return []
    text = text.replace("'", '"')
    text = re.sub(r",\s*]", "]", text)
    text = re.sub(r",\s*}", "}", text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(result, list):
        return []
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
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# =========================================================================
# SAM mask prediction
# =========================================================================

_SAM_PREDICTOR = None


def _get_sam_predictor(sam_checkpoint: str, sam_model_type: str = "vit_h"):
    """Lazy-load a SAM predictor (singleton)."""
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR
    # Ray hides GPUs from actors with num_gpus=0 by setting
    # CUDA_VISIBLE_DEVICES="".  Restore visibility so SAM can use GPU.
    _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if _cuda_vis is not None and _cuda_vis.strip() == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    from segment_anything import sam_model_registry, SamPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [SAM] Loading on device={device}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR


def _bbox_to_masks(
    predictor,
    image_rgb: np.ndarray,
    bboxes_abs: list[list[int]],
) -> list[np.ndarray]:
    """Convert a batch of bboxes to binary masks using SAM.

    Args:
        predictor: SamPredictor instance.
        image_rgb: (H, W, 3) uint8 RGB image.
        bboxes_abs: List of [x1, y1, x2, y2] in absolute pixel coords.

    Returns:
        List of (H, W) boolean masks.
    """
    if not bboxes_abs:
        return []
    predictor.set_image(image_rgb)
    device = predictor.device
    masks_out = []
    for bbox in bboxes_abs:
        box_tensor = torch.tensor([bbox], device=device, dtype=torch.float32)
        transformed = predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2])
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        masks_out.append(masks[0][0].cpu().numpy())
    return masks_out


def _mask_to_rle(mask: np.ndarray) -> dict:
    """Encode binary mask as COCO RLE."""
    from pycocotools import mask as mask_util
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# =========================================================================
# mAP computation (bbox + mask)
# =========================================================================

def _compute_det_map(
    all_pred_dets: list[list[dict]],
    all_gt_dets: list[list[dict]],
) -> dict[str, float]:
    """Compute COCO-standard mAP (bbox only, same as det CIL)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib

    gt_cat_names: set[str] = set()
    for gt_list in all_gt_dets:
        for det in gt_list:
            gt_cat_names.add(det["category"])
    if not gt_cat_names:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    all_cat_names = set(gt_cat_names)
    for pred_list in all_pred_dets:
        for det in pred_list:
            all_cat_names.add(det["category"])
    cat_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted(all_cat_names))}

    images = []
    gt_annotations = []
    ann_id = 1
    for img_idx in range(len(all_gt_dets)):
        img_id = img_idx + 1
        images.append({"id": img_id, "width": 1000, "height": 1000})
        for det in all_gt_dets[img_idx]:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            gt_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_name_to_id[det["category"]],
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": cid, "name": name} for name, cid in cat_name_to_id.items()]
    gt_dataset = {"images": images, "annotations": gt_annotations, "categories": categories}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dataset
        coco_gt.createIndex()

    dt_results = []
    for img_idx, pred_list in enumerate(all_pred_dets):
        img_id = img_idx + 1
        for det in pred_list:
            cat_id = cat_name_to_id.get(det["category"])
            if cat_id is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            dt_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "score": det.get("confidence", 1.0),
            })

    if not dt_results:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

    precision = coco_eval.eval["precision"]
    p_all = precision[:, :, :, 0, 2]
    valid = p_all > -1
    mAP = float(np.mean(p_all[valid])) if valid.any() else 0.0

    p_50 = precision[0, :, :, 0, 2]
    valid_50 = p_50 > -1
    mAP_50 = float(np.mean(p_50[valid_50])) if valid_50.any() else 0.0

    p_75 = precision[5, :, :, 0, 2]
    valid_75 = p_75 > -1
    mAP_75 = float(np.mean(p_75[valid_75])) if valid_75.any() else 0.0

    result: dict[str, float] = {"mAP": mAP, "mAP_50": mAP_50, "mAP_75": mAP_75}

    cat_ids = coco_eval.params.catIds
    id_to_name = {cid: name for name, cid in cat_name_to_id.items()}
    for k_idx, cat_id in enumerate(cat_ids):
        p_cat = precision[0, :, k_idx, 0, 2]
        valid_cat = p_cat > -1
        ap_cat = float(np.mean(p_cat[valid_cat])) if valid_cat.any() else 0.0
        cat_name = id_to_name.get(cat_id, str(cat_id))
        result[f"AP_{cat_name}"] = ap_cat

    return result


def _compute_object_det_map(
    all_pred_dets: list[list[dict]],
    all_gt_seg_dets: list[list[dict]],
    sam_actor,
    image_paths: list[str],
    image_sizes: list[tuple[int, int]],
    return_per_image: bool = False,
) -> dict[str, Any]:
    """Compute both mAP_box and mAP_mask for object detection.

    Pred bboxes → SAM → masks → COCO eval with iouType='segm'.
    GT masks from ``answer_seg`` field (polygon or RLE).

    Args:
        all_pred_dets: Per-image list of predicted dets (bbox-only, 0-1000 coords).
        all_gt_seg_dets: Per-image list of GT dets with segmentation field.
        sam_actor: SAMActor Ray handle (or None to skip mask eval).
        image_paths: Per-image absolute path for loading images for SAM.
        image_sizes: Per-image (width, height).

    Returns:
        Dict with mAP_box, mAP_box_50, mAP_mask, mAP_mask_50, etc.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_util
    import io, contextlib

    # Collect all category names
    gt_cat_names: set[str] = set()
    for gt_list in all_gt_seg_dets:
        for det in gt_list:
            gt_cat_names.add(det["category"])
    if not gt_cat_names:
        return {"mAP_box": 0.0, "mAP_box_50": 0.0, "mAP_mask": 0.0, "mAP_mask_50": 0.0}

    all_cat_names = set(gt_cat_names)
    for pred_list in all_pred_dets:
        for det in pred_list:
            all_cat_names.add(det["category"])
    cat_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted(all_cat_names))}

    n_images = len(all_gt_seg_dets)

    # Build GT dataset with segmentation
    images = []
    gt_annotations = []
    ann_id = 1
    for img_idx in range(n_images):
        img_id = img_idx + 1
        w, h = image_sizes[img_idx]
        images.append({"id": img_id, "width": w, "height": h})
        for det in all_gt_seg_dets[img_idx]:
            seg = det.get("segmentation")
            cat_id = cat_name_to_id[det["category"]]
            bbox_norm = det["bbox"]  # 0-1000

            # Convert 0-1000 bbox to absolute pixel coords for COCO format
            x1_abs = bbox_norm[0] / 1000.0 * w
            y1_abs = bbox_norm[1] / 1000.0 * h
            x2_abs = bbox_norm[2] / 1000.0 * w
            y2_abs = bbox_norm[3] / 1000.0 * h
            bw = x2_abs - x1_abs
            bh = y2_abs - y1_abs

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1_abs, y1_abs, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            }

            # Attach segmentation if available
            if seg is not None:
                if isinstance(seg, dict) and "counts" in seg:
                    # RLE format (ADE20K)
                    ann["segmentation"] = seg
                    ann["area"] = float(mask_util.area(seg))
                elif isinstance(seg, list):
                    # Polygon format (COCO)
                    ann["segmentation"] = seg
                    # Compute area from polygons
                    rle = mask_util.frPyObjects(seg, h, w)
                    if isinstance(rle, list):
                        rle = mask_util.merge(rle)
                    ann["area"] = float(mask_util.area(rle))
            gt_annotations.append(ann)
            ann_id += 1

    categories = [{"id": cid, "name": name} for name, cid in cat_name_to_id.items()]
    gt_dataset = {"images": images, "annotations": gt_annotations, "categories": categories}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dataset
        coco_gt.createIndex()

    # Build detection results (bbox)
    dt_bbox_results = []
    dt_segm_results = []
    per_image_pred_seg: list[list[dict]] = [[] for _ in range(n_images)] if return_per_image else []

    from tqdm import tqdm as _tqdm
    for img_idx, pred_list in _tqdm(enumerate(all_pred_dets), total=len(all_pred_dets), desc="SAM masks"):
        img_id = img_idx + 1
        w, h = image_sizes[img_idx]

        # Convert predicted bboxes to absolute pixel coords
        pred_bboxes_abs = []
        for det in pred_list:
            cat_id = cat_name_to_id.get(det["category"])
            if cat_id is None:
                continue
            x1 = det["bbox"][0] / 1000.0 * w
            y1 = det["bbox"][1] / 1000.0 * h
            x2 = det["bbox"][2] / 1000.0 * w
            y2 = det["bbox"][3] / 1000.0 * h
            bw, bh = x2 - x1, y2 - y1
            dt_bbox_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, bw, bh],
                "score": det.get("confidence", 1.0),
            })
            pred_bboxes_abs.append({
                "bbox_abs": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "cat_id": cat_id,
                "category": det["category"],
                "score": det.get("confidence", 1.0),
            })

        # SAM: convert bboxes to masks (via SAMActor Ray handle)
        if sam_actor is not None and pred_bboxes_abs:
            img_path = image_paths[img_idx]
            try:
                boxes = [p["bbox_abs"] for p in pred_bboxes_abs]
                rle_masks = ray.get(sam_actor.compute_masks.remote(img_path, boxes))
                if rle_masks is not None:
                    for pinfo, rle in zip(pred_bboxes_abs, rle_masks):
                        dt_segm_results.append({
                            "image_id": img_id,
                            "category_id": pinfo["cat_id"],
                            "segmentation": rle,
                            "score": pinfo["score"],
                        })
                        if return_per_image:
                            per_image_pred_seg[img_idx].append({
                                "category": pinfo["category"],
                                "segmentation": rle,
                                "score": pinfo["score"],
                            })
            except Exception as e:
                print(f"  Warning: SAM failed on {img_path}: {e}")

    result: dict[str, float] = {}

    # --- mAP_box ---
    if dt_bbox_results:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt_bbox = coco_gt.loadRes(dt_bbox_results)
            eval_bbox = COCOeval(coco_gt, coco_dt_bbox, "bbox")
            eval_bbox.evaluate()
            eval_bbox.accumulate()
        p = eval_bbox.eval["precision"]
        p_all = p[:, :, :, 0, 2]
        v = p_all > -1
        result["mAP_box"] = float(np.mean(p_all[v])) if v.any() else 0.0
        p50 = p[0, :, :, 0, 2]
        v50 = p50 > -1
        result["mAP_box_50"] = float(np.mean(p50[v50])) if v50.any() else 0.0
        p75 = p[5, :, :, 0, 2]
        v75 = p75 > -1
        result["mAP_box_75"] = float(np.mean(p75[v75])) if v75.any() else 0.0
    else:
        result["mAP_box"] = 0.0
        result["mAP_box_50"] = 0.0
        result["mAP_box_75"] = 0.0

    # --- mAP_mask ---
    if dt_segm_results:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt_segm = coco_gt.loadRes(dt_segm_results)
            eval_segm = COCOeval(coco_gt, coco_dt_segm, "segm")
            eval_segm.evaluate()
            eval_segm.accumulate()
        p = eval_segm.eval["precision"]
        p_all = p[:, :, :, 0, 2]
        v = p_all > -1
        result["mAP_mask"] = float(np.mean(p_all[v])) if v.any() else 0.0
        p50 = p[0, :, :, 0, 2]
        v50 = p50 > -1
        result["mAP_mask_50"] = float(np.mean(p50[v50])) if v50.any() else 0.0
        p75 = p[5, :, :, 0, 2]
        v75 = p75 > -1
        result["mAP_mask_75"] = float(np.mean(p75[v75])) if v75.any() else 0.0
    else:
        result["mAP_mask"] = 0.0
        result["mAP_mask_50"] = 0.0
        result["mAP_mask_75"] = 0.0

    if return_per_image:
        result["_per_image_pred_seg"] = per_image_pred_seg

    return result


def _compute_split_mask_map(
    per_image_pred_seg: list[list[dict]],
    per_image_gt_seg: list[list[dict]],
    image_sizes: list[tuple[int, int]],
    split_allowed: set[str],
    sample_gt_cats: list[set[str]],
) -> float:
    """Compute mask mAP for a category subset using pre-cached SAM masks.

    This avoids re-running SAM per task by reusing masks from the overall eval.

    Returns:
        mAP value in 0-1 scale.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_util
    import io, contextlib

    # Filter to images that have GT in the requested category set
    filtered_indices = [
        i for i, gt_cats in enumerate(sample_gt_cats)
        if gt_cats.intersection(split_allowed)
    ]
    if not filtered_indices:
        return 0.0

    # Collect category names from filtered data
    gt_cat_names: set[str] = set()
    for i in filtered_indices:
        for det in per_image_gt_seg[i]:
            cat = _normalize_label_name(str(det.get("category", "")))
            if cat in split_allowed:
                gt_cat_names.add(cat)
    if not gt_cat_names:
        return 0.0

    all_cat_names = set(gt_cat_names)
    for i in filtered_indices:
        for det in per_image_pred_seg[i]:
            cat = det.get("category", "")
            if cat in split_allowed:
                all_cat_names.add(cat)
    cat_name_to_id = {name: idx + 1 for idx, name in enumerate(sorted(all_cat_names))}

    # Build GT dataset
    images = []
    gt_annotations = []
    ann_id = 1
    for local_idx, global_idx in enumerate(filtered_indices):
        img_id = local_idx + 1
        w, h = image_sizes[global_idx]
        images.append({"id": img_id, "width": w, "height": h})
        for det in per_image_gt_seg[global_idx]:
            cat = _normalize_label_name(str(det.get("category", "")))
            if cat not in split_allowed:
                continue
            cat_id = cat_name_to_id.get(cat)
            if cat_id is None:
                continue
            seg = det.get("segmentation")
            bbox_norm = det["bbox"]
            x1_abs = bbox_norm[0] / 1000.0 * w
            y1_abs = bbox_norm[1] / 1000.0 * h
            x2_abs = bbox_norm[2] / 1000.0 * w
            y2_abs = bbox_norm[3] / 1000.0 * h
            bw = x2_abs - x1_abs
            bh = y2_abs - y1_abs
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1_abs, y1_abs, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            }
            if seg is not None:
                if isinstance(seg, dict) and "counts" in seg:
                    ann["segmentation"] = seg
                    ann["area"] = float(mask_util.area(seg))
                elif isinstance(seg, list):
                    ann["segmentation"] = seg
                    rle = mask_util.frPyObjects(seg, h, w)
                    if isinstance(rle, list):
                        rle = mask_util.merge(rle)
                    ann["area"] = float(mask_util.area(rle))
            gt_annotations.append(ann)
            ann_id += 1

    if not gt_annotations:
        return 0.0

    categories = [{"id": cid, "name": name} for name, cid in cat_name_to_id.items()]
    gt_dataset = {"images": images, "annotations": gt_annotations, "categories": categories}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dataset
        coco_gt.createIndex()

    # Build DT results from cached masks
    dt_segm_results = []
    for local_idx, global_idx in enumerate(filtered_indices):
        img_id = local_idx + 1
        for det in per_image_pred_seg[global_idx]:
            cat = det.get("category", "")
            if cat not in split_allowed:
                continue
            cat_id = cat_name_to_id.get(cat)
            if cat_id is None:
                continue
            dt_segm_results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": det["segmentation"],
                "score": det.get("score", 1.0),
            })

    if not dt_segm_results:
        return 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dt_segm_results)
        eval_segm = COCOeval(coco_gt, coco_dt, "segm")
        eval_segm.evaluate()
        eval_segm.accumulate()

    p = eval_segm.eval["precision"]
    p_all = p[:, :, :, 0, 2]
    v = p_all > -1
    return float(np.mean(p_all[v])) if v.any() else 0.0


# =========================================================================
# Data loading
# =========================================================================

def _load_categories(cat_path: str) -> list[str]:
    with open(cat_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data["categories"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Invalid categories.json format: {cat_path}")


def _list_class_names_from_jsonl(data_path: str) -> list[str]:
    cats = set()
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dets = json.loads(rec.get("answer", "[]"))
            for d in dets:
                if "category" in d:
                    cats.add(d["category"].strip().lower())
    return sorted(cats)


def _count_class_images(data_path: str, allowed_classes_norm: set[str]) -> int:
    count = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dets = json.loads(rec.get("answer", "[]"))
            for d in dets:
                if d.get("category", "").strip().lower() in allowed_classes_norm:
                    count += 1
                    break
    return count


def _filter_annotations_by_classes(
    answer_str: str, allowed_classes_norm: set[str]
) -> str:
    dets = json.loads(answer_str)
    filtered = [d for d in dets if d.get("category", "").strip().lower() in allowed_classes_norm]
    return json.dumps(filtered, ensure_ascii=False)


def _infer_example_task_id_from_answer(
    answer_str: str,
    class_to_task: dict[str, int],
) -> int:
    try:
        dets = json.loads(answer_str)
    except json.JSONDecodeError:
        return -1
    task_ids = []
    for det in dets:
        category = _normalize_label_name(det.get("category", ""))
        if category in class_to_task:
            task_ids.append(class_to_task[category])
    if not task_ids:
        return -1
    return max(task_ids)


def _count_task_images(
    data_path: str,
    class_to_task: dict[str, int],
    task_id: int,
) -> int:
    count = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            answer = rec.get("answer", "[]")
            if _infer_example_task_id_from_answer(answer, class_to_task) == task_id:
                count += 1
    return count


def _build_dataloader(
    config_data,
    tokenizer,
    processor,
    allowed_classes: Optional[Iterable[str]],
    is_train: bool,
    label_list_override: Optional[list[str]] = None,
    allowed_class_order: Optional[list[str]] = None,
    categories_path: Optional[str] = None,
    task_id_filter: Optional[int] = None,
    class_to_task: Optional[dict[str, int]] = None,
):
    data_path = config_data.train_files if is_train else config_data.val_files

    dataset = RLHFDataset(
        data_path=data_path,
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
        filter_overlong_prompts=False,
        filter_overlong_prompts_workers=config_data.filter_overlong_prompts_workers,
    )

    # ---------- task pre-filter (exemplar-replay-free) ----------
    if task_id_filter is not None:
        _tid = task_id_filter
        if class_to_task is not None:
            _class_to_task = { _normalize_label_name(k): int(v) for k, v in class_to_task.items() }
            _answer_key = config_data.answer_key
            dataset.dataset = dataset.dataset.filter(
                lambda ex, tid=_tid, answer_key=_answer_key, task_map=_class_to_task:
                    _infer_example_task_id_from_answer(ex.get(answer_key, "[]"), task_map) == tid
            )
        else:
            dataset.dataset = dataset.dataset.filter(
                lambda ex, tid=_tid: ex.get("task_id", -1) == tid
            )

    if allowed_classes is not None:
        allowed_classes_list = list(allowed_classes)
        allowed_classes_norm = {_normalize_label_name(c) for c in allowed_classes_list}

        def _has_allowed_class(example):
            answer = example.get(config_data.answer_key, "[]")
            try:
                dets = json.loads(answer)
            except json.JSONDecodeError:
                return False
            return any(
                d.get("category", "").strip().lower() in allowed_classes_norm
                for d in dets
            )

        dataset.dataset = dataset.dataset.filter(_has_allowed_class)

        def _filter_answer(example):
            example[config_data.answer_key] = _filter_annotations_by_classes(
                example[config_data.answer_key], allowed_classes_norm
            )
            # Also filter answer_seg if present
            if "answer_seg" in example and example["answer_seg"]:
                example["answer_seg"] = _filter_annotations_by_classes(
                    example["answer_seg"], allowed_classes_norm
                )
            return example

        dataset.dataset = dataset.dataset.map(_filter_answer)

        labels_for_prompt = (
            [_normalize_label_name(x) for x in label_list_override]
            if label_list_override is not None
            else [_normalize_label_name(x) for x in (allowed_class_order or allowed_classes_list)]
        )
        dataset.label_list = labels_for_prompt

    elif label_list_override is not None:
        dataset.label_list = [_normalize_label_name(x) for x in label_list_override]

    elif categories_path is not None:
        all_labels = _load_categories(categories_path)
        dataset.label_list = [_normalize_label_name(c) for c in all_labels]

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


# =========================================================================
# Trainer with object detection validation (mAP_box + mAP_mask)
# =========================================================================

class RayPPOContinualDetTrainer(RayPPOTrainer):
    """Trainer with object detection validation (mAP_box + mAP_mask)."""

    def _load_checkpoint(self) -> None:
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

        # Object detection: collect preds + GTs + seg GTs
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

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))
            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst.setdefault(key, []).extend(value)

            # Collect preds, GTs, seg GTs, and image info
            batch_labels = test_batch.non_tensor_batch.get("ground_truth")
            output_ids = test_batch.batch.get("responses")
            if output_ids is not None and batch_labels is not None:
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                # Try to get answer_seg and image paths from dataset
                batch_seg = test_batch.non_tensor_batch.get("answer_seg")
                batch_multi = test_batch.non_tensor_batch.get("multi_modal_data")

                for bi, (pred_text, gt_str) in enumerate(zip(output_texts, batch_labels)):
                    pred_answer = _extract_answer_text(pred_text)
                    pred_dets = _parse_detections(pred_answer)
                    gt_dets = _parse_detections(str(gt_str))
                    all_pred_dets.append(pred_dets)
                    all_gt_dets.append(gt_dets)
                    gt_cats = {d["category"] for d in gt_dets}
                    sample_gt_cats.append(gt_cats)

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

                    # Get actual image size from dataset metadata if available
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
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        print("Finish validation.")

        # Compute bbox-only mAP (always available)
        map_results = _compute_det_map(all_pred_dets, all_gt_dets)
        mAP_box = map_results["mAP"]

        result = {
            "val/reward_score": self.val_reward_score,
            "val/det_mAP_box": mAP_box,
            "val/det_mAP_box_50": map_results.get("mAP_50", mAP_box),
            "val/det_mAP_box_75": map_results.get("mAP_75", 0.0),
            **val_reward_metrics,
            **val_length_metrics,
        }

        # Compute mask mAP via SAM if available
        sam_ckpt = getattr(self, "sam_checkpoint", None)
        sam_type = getattr(self, "sam_model_type", "vit_h")
        has_seg_gt = any(len(seg) > 0 for seg in all_gt_seg_dets)
        run_sam = getattr(self, "_sam_during_train_val", False)
        per_image_pred_seg = None

        if sam_ckpt and has_seg_gt and run_sam:
            print("  Computing mask mAP via SAM (SAMActor)...")
            try:
                from examples.baselines.cil_det.sam_actor import SAMActor
                sam_handle = SAMActor.remote(sam_ckpt, sam_type)
                mask_results = _compute_object_det_map(
                    all_pred_dets, all_gt_seg_dets,
                    sam_handle, all_image_paths, all_image_sizes,
                    return_per_image=collect_sample_level,
                )
                ray.kill(sam_handle)
                per_image_pred_seg = mask_results.pop("_per_image_pred_seg", None)
                result["val/det_sam_mAP_mask"] = mask_results["mAP_mask"]
                result["val/det_sam_mAP_mask_50"] = mask_results["mAP_mask_50"]
                result["val/det_sam_mAP_mask_75"] = mask_results.get("mAP_mask_75", 0.0)
                # Overwrite box mAP from the full object_det eval (uses real image sizes)
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
# Snapshot helpers
# =========================================================================

def _snapshot_training_config(output_dir: str, config: Optional[PPOConfig] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd_info = {
        "script": sys.argv[0],
        "command_line": " ".join(sys.argv),
        "arguments": sys.argv[1:],
    }
    with open(os.path.join(output_dir, "train_command.json"), "w", encoding="utf-8") as f:
        json.dump(cmd_info, f, indent=2, ensure_ascii=False)
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
            print(f"[INST-SEG-CIL] Warning: --launch_script not found: {explicit_script_path}")
    if source_script is None:
        source_script = _detect_launch_script_from_parents()
    if source_script is None:
        print("[INST-SEG-CIL] Warning: launch .sh script not detected; skip script snapshot.")
        return None
    target_path = os.path.join(cil_info_dir, "train_script.sh")
    shutil.copy2(source_script, target_path)
    print(f"[INST-SEG-CIL] Saved launch script to: {target_path}")
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
        shutil.copy2(prompt_path, os.path.join(cil_info_dir, os.path.basename(prompt_path)))
    reward_spec = str(getattr(config.worker.reward, "reward_function", "") or "")
    reward_file = reward_spec.split(":", 1)[0].strip() if reward_spec else ""
    reward_path = _resolve_local_file(reward_file)
    if reward_path is not None:
        shutil.copy2(reward_path, os.path.join(cil_info_dir, os.path.basename(reward_path)))

def _build_class_order(total_classes: int, class_order: Optional[str], class_order_seed: Optional[int]) -> list[int]:
    if class_order:
        if os.path.isfile(class_order):
            with open(class_order, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "order" in data:
                data = data["order"]
            if isinstance(data, list) and all(isinstance(v, int) for v in data):
                return data
            raise ValueError("Invalid class order file format.")
        parsed = json.loads(class_order)
        if isinstance(parsed, dict) and "order" in parsed:
            parsed = parsed["order"]
        if isinstance(parsed, list) and all(isinstance(v, int) for v in parsed):
            return parsed
        raise ValueError("Invalid class order format.")
    order = list(range(total_classes))
    rng = np.random.RandomState(seed=class_order_seed)
    order = rng.permutation(order).tolist()
    return order


def _build_class_to_task_map(
    class_splits: list[list[int]],
    class_names: list[str],
) -> dict[str, int]:
    class_to_task: dict[str, int] = {}
    for task_idx, split in enumerate(class_splits):
        for class_id in split:
            class_to_task[_normalize_label_name(class_names[class_id])] = task_idx
    return class_to_task


def _resolve_task_plan(
    total_classes: int,
    base_classes: Optional[int],
    incremental_classes: Optional[int],
) -> tuple[int, int, int]:
    if incremental_classes is None:
        raise ValueError("--incremental_classes must be provided.")
    if incremental_classes <= 0:
        raise ValueError("--incremental_classes must be positive.")
    if base_classes is None:
        base_classes = incremental_classes
    if base_classes <= 0 or base_classes > total_classes:
        raise ValueError("--base_classes out of range.")
    remaining = total_classes - base_classes
    if remaining < 0 or remaining % incremental_classes != 0:
        raise ValueError("Classes cannot be split evenly.")
    total_tasks = 1 + remaining // incremental_classes
    if total_tasks < 2:
        raise ValueError("total_tasks must be >= 2.")
    return total_tasks, base_classes, incremental_classes


def _chunk_classes(class_order: list[int], base_classes: int, incremental_classes: int, total_tasks: int) -> list[list[int]]:
    splits = []
    cursor = 0
    splits.append(class_order[cursor:cursor + base_classes])
    cursor += base_classes
    for _ in range(total_tasks - 1):
        splits.append(class_order[cursor:cursor + incremental_classes])
        cursor += incremental_classes
    return splits


# =========================================================================
# Ray helpers
# =========================================================================

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
        "image_det_cil.py is a library module. Launch RaPO with "
        "python -m examples.baselines.cil_det.image_det_cil_rapo "
        "or bash scripts/det/coco/*.sh"
    )


if __name__ == "__main__":
    main()

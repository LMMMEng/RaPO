#!/usr/bin/env python3
"""Build or validate the clean COCO object detection CIL JSONL package."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_NAMES = ("train2017", "val2017", "test2017")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_image_path(image_path: str) -> str:
    path = Path(image_path)
    parts = path.parts
    for split_name in SPLIT_NAMES:
        if split_name in parts:
            split_index = parts.index(split_name)
            return str(Path(*parts[split_index:]))
    return str(Path(path.parent.name) / path.name)


def build_category_map(categories: list[dict[str, Any]]) -> tuple[dict[int, str], list[str]]:
    id_to_name = {}
    for category in sorted(categories, key=lambda item: item["id"]):
        id_to_name[int(category["id"])] = category["name"].strip().lower()
    sorted_names = [id_to_name[category_id] for category_id in sorted(id_to_name)]
    return id_to_name, sorted_names


def xywh_to_normalized_xyxy(bbox: list[float], image_width: int, image_height: int) -> list[int]:
    x_coord, y_coord, width, height = bbox
    x1 = max(0, min(1000, round(x_coord / image_width * 1000)))
    y1 = max(0, min(1000, round(y_coord / image_height * 1000)))
    x2 = max(0, min(1000, round((x_coord + width) / image_width * 1000)))
    y2 = max(0, min(1000, round((y_coord + height) / image_height * 1000)))
    return [x1, y1, x2, y2]


def process_detection_split(
    coco_data: dict[str, Any],
    image_dir: Path,
    category_id_to_name: dict[int, str],
    exclude_crowd: bool,
    min_area: float,
) -> list[dict[str, Any]]:
    image_id_to_info = {int(image["id"]): image for image in coco_data["images"]}
    image_id_to_annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for annotation in coco_data["annotations"]:
        if exclude_crowd and annotation.get("iscrowd", 0) == 1:
            continue
        if annotation.get("area", 0) < min_area:
            continue
        category_id = int(annotation["category_id"])
        if category_id not in category_id_to_name:
            continue
        image_id_to_annotations[int(annotation["image_id"])].append(annotation)

    records: list[dict[str, Any]] = []
    for image_id, annotations in sorted(image_id_to_annotations.items()):
        image_info = image_id_to_info.get(image_id)
        if image_info is None or not annotations:
            continue

        image_width = int(image_info["width"])
        image_height = int(image_info["height"])
        detections = []
        for annotation in annotations:
            category = category_id_to_name[int(annotation["category_id"])]
            bbox = xywh_to_normalized_xyxy(annotation["bbox"], image_width, image_height)
            detections.append({"category": category, "bbox": bbox})

        records.append(
            {
                "prompt": "",
                "answer": json.dumps(detections, ensure_ascii=False),
                "images": [str(image_dir / image_info["file_name"])],
                "image_width": image_width,
                "image_height": image_height,
            }
        )

    return records


def class_order(sorted_names: list[str], seed: int, mode: str) -> list[str]:
    if mode == "numpy":
        order = np.random.RandomState(seed=seed).permutation(list(range(len(sorted_names)))).tolist()
        return [sorted_names[index] for index in order]
    if mode == "python":
        names = list(sorted_names)
        random.Random(seed).shuffle(names)
        return names
    raise ValueError(f"Unsupported class order mode: {mode}")


def strict_partition(
    records: list[dict[str, Any]],
    sorted_names: list[str],
    base_classes: int,
    incremental_classes: int,
    class_order_seed: int,
    class_order_mode: str,
) -> list[dict[str, Any]]:
    ordered_classes = class_order(sorted_names, class_order_seed, class_order_mode)
    tasks = [ordered_classes[:base_classes]]
    remaining = len(ordered_classes) - base_classes
    for task_index in range((remaining + incremental_classes - 1) // incremental_classes):
        start = base_classes + task_index * incremental_classes
        end = min(start + incremental_classes, len(ordered_classes))
        tasks.append(ordered_classes[start:end])

    category_to_task = {category: task_id for task_id, task in enumerate(tasks) for category in task}
    known_up_to: dict[int, set[str]] = {}
    known: set[str] = set()
    for task_id, task in enumerate(tasks):
        known = known | set(task)
        known_up_to[task_id] = set(known)

    filtered: list[dict[str, Any]] = []
    for record in records:
        detections = json.loads(record["answer"])
        categories_in_image = {detection["category"] for detection in detections}
        task_id = max(category_to_task[category] for category in categories_in_image if category in category_to_task)
        allowed = known_up_to[task_id]
        kept_detections = [detection for detection in detections if detection["category"] in allowed]
        if not kept_detections:
            continue
        new_record = dict(record)
        new_record["answer"] = json.dumps(kept_detections, ensure_ascii=False)
        new_record["task_id"] = task_id
        filtered.append(new_record)

    return filtered


def fewshot_sample(records: list[dict[str, Any]], shots: int, seed: int) -> list[dict[str, Any]]:
    if shots <= 0:
        return records

    rng = random.Random(seed)
    category_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        detections = json.loads(record["answer"])
        for category in sorted({detection["category"] for detection in detections}):
            category_to_indices[category].append(index)

    for indices in category_to_indices.values():
        rng.shuffle(indices)

    selected: set[int] = set()
    category_counts: dict[str, int] = defaultdict(int)
    categories = sorted(category_to_indices)

    changed = True
    while changed:
        changed = False
        for category in sorted(categories, key=lambda name: (category_counts[name], name)):
            if category_counts[category] >= shots:
                continue
            while category_to_indices[category] and category_to_indices[category][0] in selected:
                category_to_indices[category].pop(0)
                changed = True
            if not category_to_indices[category]:
                continue

            selected_index = category_to_indices[category].pop(0)
            selected.add(selected_index)
            detections = json.loads(records[selected_index]["answer"])
            for covered_category in sorted({detection["category"] for detection in detections}):
                category_counts[covered_category] += 1
            changed = True

    return [records[index] for index in sorted(selected)]


def build_segmentation_index(coco_data: dict[str, Any], split_name: str) -> dict[str, list[dict[str, Any]]]:
    category_id_to_name = {int(category["id"]): category["name"].strip().lower() for category in coco_data["categories"]}
    image_id_to_info = {int(image["id"]): image for image in coco_data["images"]}
    image_to_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for annotation in coco_data["annotations"]:
        image_info = image_id_to_info.get(int(annotation["image_id"]))
        if image_info is None:
            continue
        category = category_id_to_name.get(int(annotation["category_id"]))
        if category is None:
            continue
        segmentation = annotation.get("segmentation")
        if not segmentation or isinstance(segmentation, dict):
            continue
        bbox = xywh_to_normalized_xyxy(annotation["bbox"], int(image_info["width"]), int(image_info["height"]))
        key = str(Path(split_name) / image_info["file_name"])
        image_to_annotations[key].append({"category": category, "bbox": bbox, "segmentation": segmentation})

    return image_to_annotations


def attach_segmentation(records: list[dict[str, Any]], segmentation_index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for record in records:
        image_key = normalize_image_path(record["images"][0])
        candidates = segmentation_index.get(image_key, [])
        used = [False] * len(candidates)
        segmentation_rows = []

        for detection in json.loads(record["answer"]):
            matched_index = None
            for index, candidate in enumerate(candidates):
                if used[index]:
                    continue
                if candidate["category"] == detection["category"] and candidate["bbox"] == detection["bbox"]:
                    matched_index = index
                    break
            if matched_index is None:
                raise RuntimeError(f"No polygon match for {image_key}: {detection}")
            used[matched_index] = True
            segmentation_rows.append(
                {
                    "category": detection["category"],
                    "bbox": detection["bbox"],
                    "segmentation": candidates[matched_index]["segmentation"],
                }
            )

        new_record = dict(record)
        new_record["images"] = [image_key]
        new_record["answer_seg"] = json.dumps(segmentation_rows, ensure_ascii=False)
        converted.append(new_record)

    return converted


def normalize_existing_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        new_row = dict(row)
        new_row["images"] = [normalize_image_path(image_path) for image_path in row.get("images", [])]
        normalized.append(new_row)
    return normalized


def copy_categories(source_categories: Path | None, output_categories: Path, sorted_names: list[str] | None, category_id_to_name: dict[int, str] | None) -> None:
    if source_categories is not None:
        shutil.copy2(source_categories, output_categories)
        return
    if sorted_names is None or category_id_to_name is None:
        raise ValueError("Cannot write categories.json without category metadata.")
    payload = {
        "categories": sorted_names,
        "name_to_idx": {name: index for index, name in enumerate(sorted_names)},
        "coco_id_to_name": {str(category_id): name for category_id, name in category_id_to_name.items()},
    }
    output_categories.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def task_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    for record in records:
        if "task_id" in record:
            counts[int(record["task_id"])] += 1
    return {str(task_id): counts[task_id] for task_id in sorted(counts)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean COCO object detection CIL JSONL files.")
    parser.add_argument("--coco-root", type=Path, help="COCO root with annotations/, train2017/, and val2017/.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "object_det_cil_dataset_rebuilt")
    parser.add_argument("--source-train-jsonl", type=Path, help="Existing train selection JSONL to normalize and validate.")
    parser.add_argument("--source-val-jsonl", type=Path, help="Existing validation JSONL to normalize and validate.")
    parser.add_argument("--source-categories", type=Path, help="Existing categories.json to copy.")
    parser.add_argument("--fewshot", type=int, default=5)
    parser.add_argument("--fewshot-seed", type=int, default=42)
    parser.add_argument("--base-classes", type=int, default=16)
    parser.add_argument("--incremental-classes", type=int, default=16)
    parser.add_argument("--class-order-seed", type=int, default=377)
    parser.add_argument("--class-order-mode", choices=["numpy", "python"], default="numpy")
    parser.add_argument("--min-area", type=float, default=0.0)
    parser.add_argument("--exclude-crowd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_coco = None
    val_coco = None
    category_id_to_name = None
    sorted_names = None
    segmentation_index = None

    if args.coco_root is not None:
        train_ann = args.coco_root / "annotations" / "instances_train2017.json"
        val_ann = args.coco_root / "annotations" / "instances_val2017.json"
        for required_path in [train_ann, val_ann, args.coco_root / "train2017", args.coco_root / "val2017"]:
            if not required_path.exists():
                raise FileNotFoundError(f"Required COCO path not found: {required_path}")
        train_coco = load_json(train_ann)
        val_coco = load_json(val_ann)
        category_id_to_name, sorted_names = build_category_map(train_coco["categories"])
        segmentation_index = {
            **build_segmentation_index(train_coco, "train2017"),
            **build_segmentation_index(val_coco, "val2017"),
        }

    using_existing_selection = args.source_train_jsonl is not None or args.source_val_jsonl is not None
    if using_existing_selection:
        if args.source_train_jsonl is None or args.source_val_jsonl is None:
            raise ValueError("Provide both --source-train-jsonl and --source-val-jsonl.")
        train_records = normalize_existing_rows(load_jsonl(args.source_train_jsonl))
        val_records = normalize_existing_rows(load_jsonl(args.source_val_jsonl))
        if segmentation_index is not None:
            train_records = attach_segmentation(train_records, segmentation_index)
            val_records = attach_segmentation(val_records, segmentation_index)
        build_mode = "existing_selection"
    else:
        if train_coco is None or val_coco is None or category_id_to_name is None or sorted_names is None or segmentation_index is None:
            raise ValueError("--coco-root is required when no existing selection JSONLs are provided.")
        train_records = process_detection_split(train_coco, args.coco_root / "train2017", category_id_to_name, args.exclude_crowd, args.min_area)
        train_records = strict_partition(
            train_records,
            sorted_names,
            args.base_classes,
            args.incremental_classes,
            args.class_order_seed,
            args.class_order_mode,
        )
        train_records = fewshot_sample(train_records, args.fewshot, args.fewshot_seed)
        val_records = process_detection_split(val_coco, args.coco_root / "val2017", category_id_to_name, args.exclude_crowd, args.min_area)
        train_records = attach_segmentation(train_records, segmentation_index)
        val_records = attach_segmentation(val_records, segmentation_index)
        build_mode = "raw_coco"

    write_jsonl(args.output_dir / "train_5shots.jsonl", train_records)
    write_jsonl(args.output_dir / "val.jsonl", val_records)
    copy_categories(args.source_categories, args.output_dir / "categories.json", sorted_names, category_id_to_name)

    metadata = {
        "dataset": "coco2017_object_det_cil",
        "build_mode": build_mode,
        "train_file": "train_5shots.jsonl",
        "val_file": "val.jsonl",
        "categories_file": "categories.json",
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_task_counts": task_counts(train_records),
        "image_paths": "relative_to_COCO_IMAGE_ROOT",
        "fewshot_shots": args.fewshot,
        "fewshot_seed": args.fewshot_seed,
    }
    if build_mode == "raw_coco":
        metadata.update(
            {
                "base_classes": args.base_classes,
                "incremental_classes": args.incremental_classes,
                "class_order_seed": args.class_order_seed,
                "class_order_mode": args.class_order_mode,
                "exclude_crowd": args.exclude_crowd,
                "min_area": args.min_area,
            }
        )
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

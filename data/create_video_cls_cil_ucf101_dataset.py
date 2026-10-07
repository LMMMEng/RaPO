#!/usr/bin/env python3
"""Build the UCF101 video classification CIL dataset used by the RaPO experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".webm"}


def copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        os.symlink(source.resolve(), target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def load_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_split_list(path: Path, has_label: bool) -> list[str]:
    items: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if has_label:
            relative_path = line.split()[0]
        else:
            relative_path = line.strip()
        items.append(relative_path)
    return items


def build_split(raw_root: Path, output_split: Path, split_items: list[str], class_name_map: dict[str, str], mode: str) -> int:
    video_root = raw_root / "UCF-101"
    count = 0
    for relative_item in sorted(split_items):
        raw_class_name, file_name = relative_item.split("/", 1)
        class_name = class_name_map.get(raw_class_name, raw_class_name)
        source_file = video_root / raw_class_name / file_name
        if source_file.suffix not in VIDEO_EXTENSIONS:
            continue
        copy_or_link(source_file, output_split / class_name / file_name, mode)
        count += 1
    return count


def build_fewshot_from_list(source_train: Path, target_train_5shots: Path, fewshot_list: Path, mode: str) -> int:
    count = 0
    for line in fewshot_list.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        relative_path = Path(record["relative_path"])
        copy_or_link(source_train / relative_path, target_train_5shots / relative_path, mode)
        count += 1
    return count


def count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() or path.is_symlink())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create clean UCF101 CIL split01 dataset.")
    parser.add_argument("--raw-root", type=Path, help="UCF101 raw root containing UCF-101/ and ucfTrainTestlist/")
    parser.add_argument("--standardized-source", type=Path, help="Existing standardized cil_split01 root used only to rebuild train_5shots")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "video_cls_cil_dataset")
    parser.add_argument("--manifest-root", type=Path, default=Path(__file__).resolve().parent / "video_cls_cil")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--fewshot-only", action="store_true", help="Only materialize train_5shots from the included datalist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.manifest_root / "ucf101_split01_metadata.json"
    fewshot_list_path = args.manifest_root / "fewshot_lists" / "ucf101_split01_train_5shots.jsonl"
    metadata = load_metadata(metadata_path)

    output_root = args.output_root / "UCF101" / "cil_split01"

    train_count = 0
    test_count = 0
    if not args.fewshot_only:
        if args.raw_root is None:
            raise ValueError("--raw-root is required unless --fewshot-only is set")
        split_id = int(metadata["split_id"])
        train_list = args.raw_root / "ucfTrainTestlist" / f"trainlist{split_id:02d}.txt"
        test_list = args.raw_root / "ucfTrainTestlist" / f"testlist{split_id:02d}.txt"
        train_items = read_split_list(train_list, has_label=True)
        test_items = read_split_list(test_list, has_label=False)
        train_count = build_split(args.raw_root, output_root / "train", train_items, metadata["class_name_map"], args.copy_mode)
        test_count = build_split(args.raw_root, output_root / "test", test_items, metadata["class_name_map"], args.copy_mode)

    if args.standardized_source is not None:
        source_train = args.standardized_source / "train"
    else:
        source_train = output_root / "train"
    fewshot_count = build_fewshot_from_list(source_train, output_root / "train_5shots", fewshot_list_path, args.copy_mode)

    output_metadata = {
        "dataset": "ucf101",
        "split_id": metadata["split_id"],
        "fewshot_shots": metadata["shots"],
        "fewshot_seed": metadata["seed"],
        "fewshot_source_of_truth": str(fewshot_list_path.relative_to(Path(__file__).resolve().parent.parent)),
        "copy_mode": args.copy_mode,
        "train_files_created": train_count,
        "test_files_created": test_count,
        "train_files": count_files(output_root / "train"),
        "test_files": count_files(output_root / "test"),
        "train_5shots_files": fewshot_count,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(json.dumps(output_metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_metadata, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the image classification CIL datasets used by the RaPO experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}

DATASETS = {
    "tinyimagenet": {
        "aliases": {"tin", "tiny", "tinyimagenet"},
        "folder": "tiny-imagenet-200",
        "eval_split": "val",
        "label_map": "tinyimagenet.json",
        "fewshot_list": "tinyimagenet_train_5shots.jsonl",
    },
    "cub": {
        "aliases": {"cub", "cub200"},
        "folder": "CUB_200",
        "eval_split": "test",
        "label_map": "cub.json",
        "fewshot_list": "cub_train_5shots.jsonl",
    },
    "inr": {
        "aliases": {"inr", "imagenet-r", "imagenet_r"},
        "folder": "imagenet-r",
        "eval_split": "test",
        "label_map": "imagenet_r.json",
        "fewshot_list": "imagenet_r_train_5shots.jsonl",
        "train_split_list": "imagenet_r_train.jsonl",
        "eval_split_list": "imagenet_r_test.jsonl",
    },
    "ina": {
        "aliases": {"ina", "imagenet-a", "imagenet_a"},
        "folder": "imagenet-a",
        "eval_split": "test",
        "label_map": "imagenet_a.json",
        "fewshot_list": "imagenet_a_train_5shots.jsonl",
        "train_split_list": "imagenet_a_train.jsonl",
        "eval_split_list": "imagenet_a_test.jsonl",
    },
}


def canonical_dataset_name(name: str) -> str:
    normalized = name.lower()
    for dataset_name, config in DATASETS.items():
        if normalized == dataset_name or normalized in config["aliases"]:
            return dataset_name
    raise ValueError(f"Unknown dataset: {name}")


def sanitize_label(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")


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


def image_files(directory: Path):
    for candidate in sorted(directory.iterdir()):
        if candidate.is_file() and candidate.suffix in IMAGE_EXTENSIONS:
            yield candidate


def load_label_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    label_map: dict[str, str] = {}
    for entry in entries:
        label_map[entry["source_name"]] = entry["class_name"]
        label_map[entry["class_name"]] = entry["class_name"]
        for alias in entry.get("aliases", []):
            label_map[alias] = entry["class_name"]
    return label_map


def load_label_map_class_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    names: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        class_name = entry["class_name"]
        if class_name in seen:
            continue
        seen.add(class_name)
        names.append(class_name)
    return names


def folder_names_for_class(class_name: str, label_map: dict[str, str]) -> set[str]:
    names = {class_name}
    for source_name, mapped in label_map.items():
        if mapped == class_name:
            names.add(source_name)
    return names


def ensure_class_directories(root: Path, class_names: list[str]) -> None:
    for class_name in class_names:
        (root / class_name).mkdir(parents=True, exist_ok=True)


def is_split_layout(raw_root: Path, eval_split: str) -> bool:
    return (raw_root / "train").is_dir() and (raw_root / eval_split).is_dir()


def is_flat_class_layout(raw_root: Path, label_map: dict[str, str]) -> bool:
    if is_split_layout(raw_root, "test") or is_split_layout(raw_root, "val"):
        return False
    for child in raw_root.iterdir():
        if child.is_dir() and child.name in label_map:
            return True
    return False


def load_split_records(path: Path) -> list[dict[str, str]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def build_filename_index(raw_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(raw_root):
        dirnames[:] = [name for name in dirnames if name not in {".git"}]
        for file_name in filenames:
            index.setdefault(file_name, []).append(Path(dirpath) / file_name)
    return index


def resolve_source_file(
    raw_root: Path,
    class_name: str,
    file_name: str,
    label_map: dict[str, str],
    file_index: dict[str, list[Path]],
) -> Path:
    allowed = folder_names_for_class(class_name, label_map)
    for folder in allowed:
        for base in (raw_root / "train", raw_root / "test", raw_root / "val", raw_root):
            candidate = base / folder / file_name
            if candidate.is_file() or candidate.is_symlink():
                return candidate
    hits = [path for path in file_index.get(file_name, []) if path.parent.name in allowed]
    if len(hits) == 1:
        return hits[0]
    unique_hits = file_index.get(file_name, [])
    if len(unique_hits) == 1:
        return unique_hits[0]
    raise FileNotFoundError(
        f"Could not locate {class_name}/{file_name} under {raw_root}. "
        "For ImageNet-R/A use either the cleaned train/test tree or the official Hendrycks "
        "WordNet-ID dump, together with the packaged split lists."
    )


def materialize_split_from_list(
    raw_root: Path,
    target_split: Path,
    split_list: Path,
    label_map: dict[str, str],
    mode: str,
    file_index: dict[str, list[Path]],
) -> int:
    count = 0
    for record in load_split_records(split_list):
        class_name = record["class_name"]
        file_name = record["file_name"]
        source = resolve_source_file(raw_root, class_name, file_name, label_map, file_index)
        copy_or_link(source, target_split / class_name / file_name, mode)
        count += 1
    return count


def mirror_class_folder(source_split: Path, target_split: Path, label_map: dict[str, str], mode: str) -> None:
    for source_class_dir in sorted(path for path in source_split.iterdir() if path.is_dir()):
        class_name = label_map.get(source_class_dir.name, source_class_dir.name)
        for source_file in image_files(source_class_dir):
            copy_or_link(source_file, target_split / class_name / source_file.name, mode)


def build_tinyimagenet(raw_root: Path, dataset_output: Path, label_map: dict[str, str], mode: str) -> None:
    if (raw_root / "train").is_dir() and any((raw_root / "train" / child).is_dir() for child in os.listdir(raw_root / "train")):
        sample_dir = next(path for path in sorted((raw_root / "train").iterdir()) if path.is_dir())
        if (sample_dir / "images").is_dir():
            for wnid_dir in sorted(path for path in (raw_root / "train").iterdir() if path.is_dir()):
                class_name = label_map.get(wnid_dir.name, sanitize_label(wnid_dir.name))
                for source_file in image_files(wnid_dir / "images"):
                    copy_or_link(source_file, dataset_output / "train" / class_name / source_file.name, mode)
        else:
            mirror_class_folder(raw_root / "train", dataset_output / "train", label_map, mode)

    if (raw_root / "val" / "images").is_dir() and (raw_root / "val" / "val_annotations.txt").exists():
        annotations = raw_root / "val" / "val_annotations.txt"
        for line in annotations.read_text(encoding="utf-8").splitlines():
            image_name, source_name, *_ = line.split("\t")
            class_name = label_map.get(source_name, sanitize_label(source_name))
            copy_or_link(raw_root / "val" / "images" / image_name, dataset_output / "val" / class_name / image_name, mode)
    elif (raw_root / "val").is_dir():
        mirror_class_folder(raw_root / "val", dataset_output / "val", label_map, mode)


def build_cub(raw_root: Path, dataset_output: Path, label_map: dict[str, str], mode: str) -> None:
    if all((raw_root / file_name).exists() for file_name in ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]):
        image_id_to_path = {}
        for line in (raw_root / "images.txt").read_text(encoding="utf-8").splitlines():
            image_id, relative_path = line.split(" ", 1)
            image_id_to_path[image_id] = relative_path

        class_id_to_name = {}
        for line in (raw_root / "classes.txt").read_text(encoding="utf-8").splitlines():
            class_id, raw_name = line.split(" ", 1)
            class_id_to_name[class_id] = label_map.get(raw_name, raw_name.split(".", 1)[-1])

        image_id_to_class = {}
        for line in (raw_root / "image_class_labels.txt").read_text(encoding="utf-8").splitlines():
            image_id, class_id = line.split(" ", 1)
            image_id_to_class[image_id] = class_id

        split_flags = {}
        for line in (raw_root / "train_test_split.txt").read_text(encoding="utf-8").splitlines():
            image_id, is_train = line.split(" ", 1)
            split_flags[image_id] = "train" if is_train == "1" else "test"

        for image_id, relative_path in sorted(image_id_to_path.items(), key=lambda item: item[1]):
            split_name = split_flags[image_id]
            class_name = class_id_to_name[image_id_to_class[image_id]]
            source_file = raw_root / "images" / relative_path
            copy_or_link(source_file, dataset_output / split_name / class_name / source_file.name, mode)
        return

    mirror_class_folder(raw_root / "train", dataset_output / "train", label_map, mode)
    mirror_class_folder(raw_root / "test", dataset_output / "test", label_map, mode)


def build_generic_class_dataset(
    raw_root: Path,
    dataset_output: Path,
    eval_split: str,
    label_map: dict[str, str],
    mode: str,
    train_split_list: Path | None = None,
    eval_split_list: Path | None = None,
) -> None:
    if is_split_layout(raw_root, eval_split):
        mirror_class_folder(raw_root / "train", dataset_output / "train", label_map, mode)
        mirror_class_folder(raw_root / eval_split, dataset_output / eval_split, label_map, mode)
        return

    if train_split_list is None or eval_split_list is None:
        raise ValueError(
            f"{raw_root} is not a cleaned train/{eval_split} tree and no packaged split lists were provided."
        )
    if not train_split_list.exists() or not eval_split_list.exists():
        raise FileNotFoundError(
            f"Missing packaged split lists:\n  {train_split_list}\n  {eval_split_list}"
        )
    if not is_flat_class_layout(raw_root, label_map):
        raise ValueError(
            f"{raw_root} is not a recognized ImageNet-R/A layout.\n"
            f"Expected either:\n"
            f"  1) cleaned split: {raw_root}/train/<class_or_wnid>/ and {raw_root}/{eval_split}/<class_or_wnid>/\n"
            f"  2) official Hendrycks dump: {raw_root}/<wnid>/*.jpg"
        )

    file_index = build_filename_index(raw_root)
    materialize_split_from_list(raw_root, dataset_output / "train", train_split_list, label_map, mode, file_index)
    materialize_split_from_list(raw_root, dataset_output / eval_split, eval_split_list, label_map, mode, file_index)


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
    parser = argparse.ArgumentParser(description="Create clean image classification CIL datasets.")
    parser.add_argument("--dataset", required=True, help="tinyimagenet, cub, inr, or ina")
    parser.add_argument("--raw-root", type=Path, help="Extracted raw dataset root")
    parser.add_argument("--standardized-source", type=Path, help="Existing standardized dataset root used only to rebuild train_5shots")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "image_cls_cil_dataset")
    parser.add_argument("--manifest-root", type=Path, default=Path(__file__).resolve().parent / "image_cls_cil")
    parser.add_argument("--copy-mode", choices=["copy", "symlink", "hardlink"], default="symlink")
    parser.add_argument("--fewshot-only", action="store_true", help="Only materialize train_5shots from the included datalist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = canonical_dataset_name(args.dataset)
    config = DATASETS[dataset_name]
    dataset_output = args.output_root / config["folder"]
    label_map_path = args.manifest_root / "label_maps" / config["label_map"]
    fewshot_list_path = args.manifest_root / "fewshot_lists" / config["fewshot_list"]

    label_map = load_label_map(label_map_path)
    class_names = load_label_map_class_names(label_map_path)
    train_split_list = None
    eval_split_list = None
    if config.get("train_split_list"):
        train_split_list = args.manifest_root / "split_lists" / config["train_split_list"]
    if config.get("eval_split_list"):
        eval_split_list = args.manifest_root / "split_lists" / config["eval_split_list"]

    if not args.fewshot_only:
        if args.raw_root is None:
            raise ValueError("--raw-root is required unless --fewshot-only is set")
        if dataset_name == "tinyimagenet":
            build_tinyimagenet(args.raw_root, dataset_output, label_map, args.copy_mode)
        elif dataset_name == "cub":
            build_cub(args.raw_root, dataset_output, label_map, args.copy_mode)
        else:
            build_generic_class_dataset(
                args.raw_root,
                dataset_output,
                config["eval_split"],
                label_map,
                args.copy_mode,
                train_split_list=train_split_list,
                eval_split_list=eval_split_list,
            )

    if args.standardized_source is not None:
        source_train = args.standardized_source / "train"
    else:
        source_train = dataset_output / "train"
    selected_count = build_fewshot_from_list(source_train, dataset_output / "train_5shots", fewshot_list_path, args.copy_mode)
    if class_names:
        ensure_class_directories(dataset_output / "train_5shots", class_names)
        if (dataset_output / "train").exists():
            ensure_class_directories(dataset_output / "train", class_names)
        eval_dir = dataset_output / config["eval_split"]
        if eval_dir.exists():
            ensure_class_directories(eval_dir, class_names)

    metadata = {
        "dataset": dataset_name,
        "folder": config["folder"],
        "fewshot_shots": 5,
        "fewshot_seed": 42,
        "fewshot_source_of_truth": str(fewshot_list_path.relative_to(Path(__file__).resolve().parent.parent)),
        "copy_mode": args.copy_mode,
        "label_map_classes": len(class_names),
        "train_5shots_classes": len([p for p in (dataset_output / "train_5shots").iterdir() if p.is_dir()]) if (dataset_output / "train_5shots").exists() else 0,
        "train_5shots_files": selected_count,
        "train_files": count_files(dataset_output / "train"),
        "eval_split": config["eval_split"],
        "eval_files": count_files(dataset_output / config["eval_split"]),
    }
    dataset_output.mkdir(parents=True, exist_ok=True)
    (dataset_output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

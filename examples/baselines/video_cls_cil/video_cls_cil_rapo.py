"""Video classification CIL trainer – persistent implementation with persistent workers.

Thin wrapper around image_cls_cil_rapo.py. Patches the sample
counting function to use video extensions. All CIL logic (persistent
workers, in-memory weight copy, optimizer carry-over) is inherited exactly.
"""

import os

# Patch _count_class_samples before importing persistent implementation
import examples.baselines.img_cls_cil.image_cls_cil as base_cil

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _count_class_samples_video(data_path: str, allowed_classes_norm: set[str]) -> int:
    if not os.path.isdir(data_path):
        raise ValueError(f"data_path '{data_path}' is not a directory; only folder-per-class datasets are supported in CIL mode.")
    total = 0
    for class_name in os.listdir(data_path):
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        norm = base_cil._normalize_label_name(class_name)
        if norm not in allowed_classes_norm:
            continue
        for fname in os.listdir(class_dir):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                total += 1
    return total


base_cil._count_class_samples = _count_class_samples_video

import examples.baselines.img_cls_cil.image_cls_cil_rapo as image_cil_rapo


def main() -> None:
    image_cil_rapo.main()


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    main()

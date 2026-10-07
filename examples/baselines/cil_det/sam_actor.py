"""sam_actor.py
===============================================================================
Standalone Ray actor for SAM (Segment Anything Model) inference.

This actor deliberately does **NOT** import any verl / vllm modules so that
CUDA initialises cleanly — solving the problem where ``from vllm import LLM``
(triggered by verl imports) caches ``device_count=0`` in the C++ runtime when
the parent Ray actor has ``CUDA_VISIBLE_DEVICES=""``.

Usage (from within a Runner / Trainer that *does* import verl)::

    from examples.baselines.cil_det.sam_actor import SAMActor

    sam = SAMActor.remote(checkpoint_path, "vit_h")
    rle_masks = ray.get(sam.compute_masks.remote(image_path, bboxes_abs))
    ray.kill(sam)
"""

from __future__ import annotations

import os
from typing import Optional

import ray


@ray.remote(num_cpus=1)
class SAMActor:
    """Ray actor that loads SAM on GPU and converts bboxes → RLE masks.

    All heavy imports (torch, cv2, segment_anything, pycocotools) are deferred
    to ``__init__`` so that the CUDA environment variable can be fixed first.
    """

    def __init__(self, sam_checkpoint: str, sam_model_type: str = "vit_h", sam_dtype: str = "bf16"):
        # Ray sets CUDA_VISIBLE_DEVICES="" for actors with num_gpus=0.
        # Pop it BEFORE importing torch so CUDA can see GPUs.
        _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if _cuda_vis is not None and _cuda_vis.strip() == "":
            os.environ.pop("CUDA_VISIBLE_DEVICES")

        import torch
        import numpy as np  # noqa: F401 — cached for later use
        from segment_anything import sam_model_registry, SamPredictor

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve dtype for mixed-precision inference
        _dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._amp_dtype = _dtype_map.get(sam_dtype, torch.bfloat16)
        self._use_amp = self._device == "cuda" and self._amp_dtype != torch.float32

        print(
            f"  [SAM Actor] Loading SAM ({sam_model_type}) on device={self._device}  "
            f"dtype={sam_dtype}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}"
        )
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self._device)
        self._predictor = SamPredictor(sam)

    # ------------------------------------------------------------------
    def ping(self) -> dict:
        """Health-check: returns device info."""
        return {
            "device": str(self._device),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"),
        }

    # ------------------------------------------------------------------
    def compute_masks(
        self,
        image_path: str,
        bboxes_abs: list[list[int]],
    ) -> Optional[list[dict]]:
        """Run SAM on *one* image and return RLE-encoded masks.

        Args:
            image_path: Absolute path to the image file.
            bboxes_abs: List of ``[x1, y1, x2, y2]`` in absolute pixel coords.

        Returns:
            List of COCO-RLE dicts (one per bbox), or ``None`` on failure.
        """
        if not bboxes_abs:
            return []

        import cv2
        import numpy as np
        from pycocotools import mask as mask_util

        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                print(f"  [SAM Actor] Warning: could not load image {image_path}")
                return None
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"  [SAM Actor] Warning: image read failed {image_path}: {e}")
            return None

        torch = self._torch
        predictor = self._predictor
        device = self._device
        h, w = image_rgb.shape[:2]

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp,
        ):
            predictor.set_image(image_rgb)
            rle_results: list[dict] = []
            for bbox in bboxes_abs:
                box_tensor = torch.tensor([bbox], device=device, dtype=torch.float32)
                transformed = predictor.transform.apply_boxes_torch(box_tensor, image_rgb.shape[:2])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=False,
                )
                mask_np = masks[0][0].cpu().numpy()
                rle = mask_util.encode(np.asfortranarray(mask_np.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                rle["size"] = [h, w]
                rle_results.append(rle)

        return rle_results

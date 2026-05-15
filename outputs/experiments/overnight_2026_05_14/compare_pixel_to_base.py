"""Compare pixel_to_base_xy calibration matrices across pixel_delta heads.

The matrix is the linear map fit by `_fit_pixel_to_base_xy` from
`delta_port_minus_plug_px` (center camera) to `delta_port_minus_plug_m`
(base frame xy). It's used at runtime by RunACT.py to convert the head's
predicted pixel delta into the base-frame xy correction.

If the new long-range matrix is wildly different from the old one (sign
flips, magnitude >2x), the ASSIST-mode runtime behavior won't be what
F-safe was tuned for — and that's a silent foot-gun the Plan agent
specifically flagged.
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch


def load_calib(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    return {
        "name": str(ckpt_path),
        "target_scale": cfg.get("target_scale"),
        "image_width": cfg.get("image_width"),
        "image_height": cfg.get("image_height"),
        "epoch": ckpt.get("epoch"),
        "pixel_to_base_xy": np.array(cfg["pixel_to_base_xy"], dtype=np.float64),
        "metrics": ckpt.get("metrics", {}),
    }


def main():
    candidates = [
        Path("/home/saivemu/code/aic/outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2/best_visual_servo.pt"),
        Path("/home/saivemu/code/aic/outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2_e60/best_visual_servo.pt"),
        Path("/home/saivemu/code/aic/outputs/experiments/vision_servo_labels/models/visual_pixel_delta_longrange40_o80_e60/best_visual_servo.pt"),
    ]
    labels = [
        "OLD shipped  (10 ep, balanced20_o25)",
        "OLD control  (60 ep, balanced20_o25)",
        "NEW G-lite   (60 ep, longrange40_o80)",
    ]
    calibs = []
    for p, label in zip(candidates, labels):
        if not p.exists():
            print(f"  [skip] {label}: {p} not found yet")
            continue
        c = load_calib(p)
        c["label"] = label
        calibs.append(c)
        print(f"\n{label}")
        print(f"  path: {p}")
        print(f"  best epoch: {c['epoch']}")
        print(f"  target_scale: {c['target_scale']}")
        print(f"  image size: {c['image_width']}x{c['image_height']}")
        print(f"  val_mae_xy_norm_px: {c['metrics'].get('mae_xy_norm_px', '?')}")
        print(f"  within_5px: {c['metrics'].get('within_5px', '?')}")
        print(f"  within_10px: {c['metrics'].get('within_10px', '?')}")
        print(f"  pixel_to_base_xy (m/px):")
        M = c["pixel_to_base_xy"]
        print(f"    base_x = {M[0, 0]: .3e} * dpx + {M[0, 1]: .3e} * dpy + {M[0, 2]: .3e}")
        print(f"    base_y = {M[1, 0]: .3e} * dpx + {M[1, 1]: .3e} * dpy + {M[1, 2]: .3e}")
        # Effective scale: how many mm of base motion per pixel of image error?
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        print(f"    determinant: {det: .3e}")
    if len(calibs) < 2:
        return
    print("\n=== Pairwise ratios (mostly want them ~1; large = significant change) ===")
    base = calibs[0]["pixel_to_base_xy"]
    for c in calibs[1:]:
        M = c["pixel_to_base_xy"]
        # Compare on the linear part (drop the bias column)
        ratios = M[:, :2] / np.where(np.abs(base[:, :2]) < 1e-9, np.nan, base[:, :2])
        print(f"\n  {c['label']} / {calibs[0]['label']}:")
        for i in range(2):
            for j in range(2):
                rij = ratios[i, j]
                tag = ""
                if np.isnan(rij):
                    tag = " (base near-zero)"
                elif abs(rij) > 3 or abs(rij) < 0.33:
                    tag = " <-- WARN: >3x deviation"
                elif rij < 0:
                    tag = " <-- WARN: sign flipped"
                print(f"    [{i},{j}] ratio = {rij: .3f}{tag}")


if __name__ == "__main__":
    main()

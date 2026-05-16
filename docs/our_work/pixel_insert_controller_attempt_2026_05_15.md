# Pixel-Gated Insert Controller Attempt - 2026-05-15

Goal: add a fast hybrid final-stage controller that uses the existing
`pixel_delta` visual-servo head for xy alignment, then performs a guarded
high-stiffness descent only when the predicted xy error is small and stable.

## Implementation

Added an env-gated `PI_*` state machine to `RunACT.py`.

- `PI_ALIGN`: drive xy from visual-servo port-minus-plug error.
- `PI_DESCEND`: reset pose target to current TCP, keep xy correction active,
  and descend with boosted z stiffness.
- `PI_INSERT`: after contact, freeze xy and continue downward to a depth target.
- `PI_COMPLETE`: optionally return early for duration bonus.

Also added `AIC_FINAL_HELPER_PLUG_TYPES` so final-stage helpers can be gated to
specific plug families.

The controller is disabled by default.

## Local Scores

Base policy:
`outputs/train/act_final_recovery30_scscaled_clean25_v1/checkpoints/010000/pretrained_model`

Visual model:
`outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2/best_visual_servo.pt`

| Run | Total | Notes |
| --- | ---: | --- |
| `score_pixel_insert_balanced_start11_v1` | 148.155 | Trial 2 partial insertion; 0.75 below current best. |
| `score_pixel_insert_noassist_start11_v1` | 139.516 | Removing continuous VS assist hurt trial 2. |
| `score_pixel_insert_sfp_only_start11_v1` | 126.955 | SFP-only task gate did not reproduce the partial insertion. |
| `score_pixel_insert_aggressive_descend_v1` | 145.928 | Trial 1 partial insertion; extra descent hurt other trials. |

## Decision

Do not submit this yet. It is close enough to keep as a useful controller
scaffold, but none of the fast variants beat the current best local score
`148.903` from the ACT-only checkpoint image.

The result confirms that controller/hybrid finishing can create partial
insertions, but the existing pixel-delta head is not stable enough across all
three trials. Better port localization remains the likely next high-leverage
piece.

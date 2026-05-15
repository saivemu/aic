# Resume state — 2026-05-14 evening (workstation move)

Pinned context for picking the work back up after the workstation power-off.
Look at this first when you're back online.

## Currently in flight (as of 18:43 PT)

| Job | State | Output | ETA |
|---|---|---|---|
| Long-range dataset recording | **DONE** | `outputs/experiments/vision_servo_labels/data/vision_servo_longrange40_o80/` (40 ep, 15,672 frames, median xy 33 mm, p90 86 mm, max 251 mm) | finished 18:25 |
| Diffusion Policy training (`Plan G v1`) | **running** at step 32k / 40k, loss ~0.001 | `outputs/train/diffusion_plan_g_v1/checkpoints/{010000,020000,030000,last}` | ~20 min |
| pixel_delta head retrain on `longrange40_o80` | **queued** | `outputs/experiments/vision_servo_labels/models/visual_pixel_delta_longrange40_o80/` | starts when GPU frees |
| Compose-eval Plan G + new pixel_delta in ASSIST mode | **queued** | `outputs/experiments/overnight_2026_05_14/results/plan_g_*_summary.txt` | starts after pixel_delta train |

When Plan G training completes, the background runner finalizes a
`training_complete.txt` and a commit/push, so the remote will reflect the
final state automatically.

## Open decision for you (after the workstation move)

**Plan G vs Plan H — which direction after the in-flight jobs land?**

- **Plan G (current track)**: Diffusion-on-`aic_act_v2` + pixel_delta-on-`longrange40_o80`, composed at runtime via ASSIST + z-stiff. Tests whether replacing ACT with a multimodal policy and replacing the perception head with a long-range one lifts us past the 127.77 mean. Local eval gates tonight; ship if ≥ Plan F-safe (127.77) +5.
- **Plan H (Phase 2 if Plan G plateaus)**: re-record `aic_act_v2++` in LeRobot format (`record_dataset.py`, 300 ep, `AIC_CHEATCODE_XY_OFFSET_MAX_M=0.05`) so the policy itself sees the long-range distribution Plan D never trained on. ~5 hr collection + ~2 hr Diffusion training, but unblocks the policy-side coverage gap. Plan G's pixel_delta head can be reused as-is.

## What's where

### Data
- `~/.cache/huggingface/lerobot/saivemu/aic_act_v2/` — 299-ep Plan D training set (43-D state, 512×576 images). Used for Plan G Diffusion.
- `outputs/experiments/vision_servo_labels/data/vision_servo_longrange40_o80/` — new long-range visual-servo dataset (15,672 frames, median 33 mm). Used for the new pixel_delta head. JSONL + JPEG format (not LeRobot).

### Models / checkpoints
- `outputs/train/diffusion_plan_g_v1/checkpoints/last` → currently symlinks to 030000; will point to 040000 when training finishes
- `outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2/best_visual_servo.pt` — the OLD pixel_delta head shipped in `plane-pixel-v1` and used in `assist-pixel-zstiff-v1`
- (queued) `outputs/experiments/vision_servo_labels/models/visual_pixel_delta_longrange40_o80/best_visual_servo.pt`

### ECR (already pushed, ready to submit anytime)
- `…/bot-squad-l2-learning-loop:plan-d-v1` — live submission, 123.06
- `…/bot-squad-l2-learning-loop:assist-pixel-zstiff-v1` — **+4.7 over Plan D, recommended**, 5-run mean 127.77
- `…/bot-squad-l2-learning-loop:plane-pixel-v1` — high-variance, max 139.02 with one partial insertion

## Resume commands

After the move, before doing anything else:

```bash
# 1. Verify training finished cleanly
ls outputs/train/diffusion_plan_g_v1/checkpoints/
cat outputs/experiments/overnight_2026_05_14/logs/training_complete.txt
# Expected: 010000, 020000, 030000, 040000, last (-> 040000)

# 2. Train new pixel_delta head on long-range data (~30 min)
pixi run --as-is python aic_utils/lerobot_robot_aic/scripts/train_visual_servo.py \
  --data-root outputs/experiments/vision_servo_labels/data/vision_servo_longrange40_o80 \
  --output outputs/experiments/vision_servo_labels/models/visual_pixel_delta_longrange40_o80 \
  --target pixel_delta \
  --camera center \
  --epochs 60 \
  --batch-size 64 \
  --max-xy-target-m 0.20 \
  --min-xy-target-m 0.0

# 3. Build a Plan G image and eval against Plan F-safe (127.77 baseline)
# (Dockerfile and compose overlay templates are noted in the
#  overnight_2026_05_14_progress.md ship recommendation; copy
#  Dockerfile.assist_pixel_zstiff as a starting point, swap in the new
#  pixel_delta model file + the Plan G Diffusion checkpoint.)
```

## Daily ECR slot status

Plan D-v1 is still live. Plans F-safe (assist-pixel-zstiff-v1) and F-aggressive
(plane-pixel-v1) are in ECR untouched. No submission slot has been spent on
the new plans yet — the call is yours after Plan G eval lands.

## Files modified this session (already committed and pushed)

- Plan F runtime: `aic_example_policies/aic_example_policies/ros/RunACT.py`, `docker/docker-compose.yaml`
- Plan F image recipes: `docker/aic_model/Dockerfile.assist_pixel_zstiff`, `docker/aic_model/Dockerfile.planE_pixel`
- Overnight log: `docs/overnight_2026_05_14_progress.md`
- Status tracker: `docs/status.md` (Plan F-safe and F-aggressive rows added)
- Experiment artifacts: `outputs/experiments/overnight_2026_05_14/*` (env files, compose overlays, run script, summaries)

## What was actually disconnected for the move

- Network: safe to disconnect any time. Nothing reaches out — dataset is local cache, no ECR/HF uploads in flight.
- GPU: in use by Diffusion training until ~19:05 PT. Wait for the "training complete" notification, or kill the train and resume from `030000` (loses ~5 min of training, recoverable).

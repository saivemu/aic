# Visual Servo Experiment Log

Date: 2026-05-14

Goal: push the Plan D score from the 120s toward 150-200 by solving the final
alignment/insertion stage without violating the challenge rule boundary. Runtime
controllers must use only legal observations. Training-time TF labels are allowed
only for dataset creation and supervision.

## Current Result

The best legal local eval from this round is still proximity-only:

| Variant | Total | Contact/insertion | Notes |
|---|---:|---|---|
| Plan D shipped baseline | 123.06-123.58 | none | Current ECR submission baseline. |
| Visual servo, direct action regression | 118.57 | none | Regressed xy action collapsed near zero at runtime. |
| Visual servo, xy direction classifier | 124.85 | none | Best local score so far. Nonzero xy corrections, but no contact/insertion. |
| Visual servo, pixel delta regressor | 124.30 | none | Similar to direction, weak y accuracy. |
| xy direction with 45 s trial cap | 112.56 | none | Extra time hurt duration score badly. |

The important correction to the earlier approach: score is not just final
accuracy. The rubric rewards stage speed. A long final search can score lower
than doing nothing unless it reliably creates contact/insertion. Keep the fast
Plan D approach and spend only a short, late final-stage budget unless there is
strong evidence the controller can finish the insert.

## What Changed In Code

### Training-time alignment enrichment

`aic_example_policies/aic_example_policies/ros/CheatCode.py` now has
environment-controlled training-only enrichment:

- lateral offset injection before final insertion
- optional xy dither during final descent
- configurable xy integral gain and windup
- configurable descent step size

These options are for generating harder and more useful alignment labels. They
must not be used as runtime ground-truth assistance in a scored submission.

### TF-labeled visual servo dataset recorder

`aic_utils/lerobot_robot_aic/scripts/record_visual_servo_dataset.py` records
legal observation images/state/action while using training-time TF to save labels:

- projected port and plug pixels per camera
- base-frame `delta_port_minus_plug_m`
- 43-D state, task identity, and latest action
- manifest plus JSONL labels and saved JPEG frames

The intended dataset shape is small and final-stage-heavy, not another broad
full-episode imitation dataset.

### Small visual-servo trainer

`aic_utils/lerobot_robot_aic/scripts/train_visual_servo.py` trains a compact
image+state model from the recorded labels. Supported target modes:

- `delta`: base-frame port-minus-plug xyz delta
- `action_linear`: recorded linear xyz action
- `xy_direction`: discrete x/y negative-hold-positive classes
- `pixel_delta`: center-image port-minus-plug pixel delta plus fitted pixel-to-base calibration

The direction classifier was added because continuous regression kept collapsing
on tiny, symmetric xy labels.

### Optional runtime controllers in RunACT

`aic_example_policies/aic_example_policies/ros/RunACT.py` keeps Plan D as the
default, then optionally enables one of several final-stage controllers:

- final spiral/search controller
- insertion-only ACT checkpoint handoff
- learned visual servo controller
- configurable 30 s trial cap via `AIC_MAX_TRIAL_S`

The default path is still Plan D behavior unless the new environment variables
are set.

### Eval utility

`aic_utils/lerobot_robot_aic/scripts/eval_checkpoints.py` now handles zero
normalization denominators and supports `--tail-frames`, so final-stage error can
be evaluated separately from full-episode average error.

## Local Data And Model Artifacts

The datasets and model checkpoints are local under `outputs/` and are not
committed. The small replay/record compose wrappers in
`outputs/experiments/vision_servo_labels/` are committed for reproducibility.

| Path | Purpose |
|---|---|
| `outputs/experiments/vision_servo_labels/data/vision_servo_balanced20_o25_s2` | Main 20-episode alignment-labeled dataset. |
| `outputs/experiments/vision_servo_labels/models/visual_action_balanced20_o25_s2` | Direct action regressor. |
| `outputs/experiments/vision_servo_labels/models/visual_delta_balanced20_o25_s2` | Base-frame delta regressor. |
| `outputs/experiments/vision_servo_labels/models/visual_direction_balanced20_o25_s2` | Best xy direction classifier. |
| `outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2` | Pixel delta regressor with fitted calibration. |

Main dataset stats:

- 20 requested episodes, 19 with labels
- 8645 labeled frames and images
- balanced across SFP and SC targets
- final xy target norm median about 4.3 mm, p90 about 13.7 mm

## Best Reproducible Local Command

The current best local visual-servo run used the direction model with a short
late takeover and no trial extension:

```bash
AIC_POLICY_PLAN=d \
AIC_VISUAL_SERVO_START_S=9.0 \
AIC_VISUAL_SERVO_Z_MODE=act \
AIC_VISUAL_SERVO_DIRECTION_SPEED_MPS=0.003 \
AIC_VISUAL_SERVO_MAX_XY_SPEED_MPS=0.006 \
docker compose \
  -f docker/docker-compose.yaml \
  -f docker/docker-compose.override.yaml \
  -f outputs/experiments/vision_servo_labels/docker-compose.eval-visual-direction.yaml \
  up --abort-on-container-exit --exit-code-from eval
```

Score:

- trial 1: 42.59
- trial 2: 42.51
- trial 3: 39.75
- total: 124.85

This is not yet a submission-worthy jump over Plan D because it has no insertion
or contact bonus.

## Lessons From The Failed Branches

1. Plain continuous regression is not enough for final xy.

   `delta` and `action_linear` models can look reasonable on validation but still
   produce near-zero corrections in closed loop. This matches the ACT collapse
   diagnosis: tiny, symmetric final xy labels make the median/mean look safe,
   while the task needs millimeter-correct directional behavior.

2. Direction classification helps collapse but does not solve insertion.

   `xy_direction` produced real nonzero correction commands and the best score
   of the round, but it still did not make contact or insertion. It needs a
   confidence gate and probably should assist ACT xy rather than replace ACT xy.

3. Center-camera pixel delta is promising but underpowered.

   Pixel delta was close to direction score but had high y error. Side cameras or
   a true keypoint/heatmap detector are likely better than one global center-image
   regression head.

4. More time is expensive.

   Increasing the trial cap to 45 s dropped score to 112.56. Do not spend extra
   wall-clock time unless the final controller has high contact/insertion odds.

5. Training labels need to be alignment-rich, not just more numerous.

   The useful data is final-stage, offset-rich, successful or partial-insertion
   behavior. More broad approach data repeats what Plan D already handles.

## Recommended Next Order

1. Make the visual servo speed-preserving.

   Add confidence-gated assist mode: keep Plan D's action as the default and add
   only a capped xy correction when the model is confident. Do not replace ACT xy
   unconditionally. Keep `AIC_MAX_TRIAL_S=30.0`.

2. Add a short guarded finish, not a long search.

   Once visual error is small and confidence is high, spend at most a few seconds
   on a slow downward push with force backoff. If confidence is low, let Plan D
   finish normally.

3. Collect a better final-stage dataset.

   Use the CheatCode enrichment to create alignment-rich episodes with deliberate
   lateral offsets, slower final descent, both SFP/SC variants, and only successful
   or partial-insertion final windows. Include side camera labels.

4. Replace global regression with keypoints/heatmaps.

   Train port and plug keypoint detectors from the TF projections, then servo from
   visible keypoint offsets. This should be more data-efficient than regressing
   tiny Cartesian velocities directly.

5. Submit only when score clears variance.

   Use Plan D as the fallback. A final-alignment branch needs at least a stable
   +3 to +5 local gain before burning a submission slot, and a real >150 path
   probably requires contact/insertion in at least one trial.

# Overnight 2026-05-14 progress log

## Hand-off banner (read this first when you wake up)

**Two ECR-ready images are built and verified locally.** Pick which to push:

| Image tag | Config | Local mean | Local max | Variance | Risk |
|---|---|---:|---:|---|---|
| `aic-runact:plane-pixel-v1` | pixel_delta REPLACE | 124.5 / 138.7* | 139.0 | high (28 pt range) | could regress to ~110 |
| `aic-runact:assist-pixel-zstiff-v1` | pixel_delta ASSIST + z-stiff 500 | **127.77** | 128.41 | **tight 1.16 pt** | safe +4.7 over Plan D |

*138.7 is the score from this morning's verification run of the baked image;
the original 3-run mean was 124.5 with one trial hitting partial insertion.

Both images are self-contained — config baked into a custom entrypoint, so
they don't depend on compose env overrides. The cluster can `docker run` them
directly. Both verified locally with our compose stack to the score above.

**Recommendation: push `assist-pixel-zstiff-v1`.** Tight variance + clear
+4.7 pts over Plan D = expected leaderboard improvement. The plane-pixel
image has a higher ceiling but its lower bound (110) is *worse* than Plan D.

Did **NOT** hit 200. No reliable port insertion this round. The blocker is
port localization at the 50–90 mm range where Plan D leaves the gripper —
the existing classifier/pixel_delta heads only have signal at ≤14 mm
offsets. Real progress past ~140 needs port-keypoint perception (see
"What I think is needed" further down).

## ECR images (pushed 2026-05-14 ~03:13 PT)

Both images are **already in ECR** and ready to paste into the submission
portal whenever you want to use a daily slot. Plan D shipped (123.06) stays
the live submission until you change it.

| Submission portal URI | Local-eval result | Risk |
|---|---:|---|
| `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:assist-pixel-zstiff-v1` | 128.28 verified, 5-run mean 127.77, spread 1.16 pt | **low — recommended** |
| `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:plane-pixel-v1` | 138.67 verified, 3-run range 110–139 | high — could regress |

Digests for the record:
- `assist-pixel-zstiff-v1` → `sha256:db5fd50759d737718608f3d65ed6757741d335eaaeefe26af7b050daf42c41f4`
- `plane-pixel-v1` → `sha256:6cfdce784add64d55c886e8156e9ee7e496165086d0d09ed73d414385a764141`

Tags are immutable in this ECR repo, so neither can be overwritten — if you
need a follow-up build, bump to `:assist-pixel-zstiff-v2` etc.

**To submit:** copy one of the URIs above and paste into the submission portal
`OCI Image` field. Pushing the image alone does NOT trigger evaluation; the
portal entry is what does.

**Re-authenticate if needed** (ECR tokens last 12 h):
```bash
aws --profile bot-squad-l2-learning-loop ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 973918476471.dkr.ecr.us-east-1.amazonaws.com
```

## Verification commands

```bash
# Re-run either image locally to confirm before push:
# (each takes ~2 minutes)

# assist-pixel-zstiff
docker compose \
  -f docker/docker-compose.yaml \
  -f docker/docker-compose.override.yaml \
  -f outputs/experiments/overnight_2026_05_14/docker-compose.use-assist-pixel-zstiff.yaml \
  up --abort-on-container-exit --exit-code-from eval

# plane-pixel
docker compose \
  -f docker/docker-compose.yaml \
  -f docker/docker-compose.override.yaml \
  -f outputs/experiments/overnight_2026_05_14/docker-compose.use-plane-pixel.yaml \
  up --abort-on-container-exit --exit-code-from eval
```

The trick that made the images self-configuring: a custom entrypoint in each
Dockerfile that `export`s the desired AIC_VISUAL_SERVO_* env vars *after*
compose has applied its own. This makes the image immune to the base
compose's `${VAR:-}` empty-string overrides.

## TL;DR

**New best: 127.77 mean / 128.41 max over 5 compose runs (+4.7 over Plan D).**
Config: ASSIST mode + `pixel_delta` visual-servo head + xy speed cap 10 mm/s
+ z-stiffness override 500 N/m during VS_ASSIST. All env-var-driven, no model
retrain, no new data needed. Defaults preserve Plan D / Plan E behavior. See
"Ship recommendation" at the bottom for the exact command.

The breakthrough came from switching the visual-servo head from `xy_direction`
(direction classifier, used by Plan E) to `pixel_delta` (continuous pixel
regressor) AND running it in ADD-to-ACT (ASSIST) mode AND boosting z stiffness
so the gripper actually descends toward the port. Plan E REPLACE mode with the
pixel_delta head *can* hit 139 (partial insertion in trial 2) but is
high-variance (110–139). ASSIST mode trades that ceiling for tight ~1 pt
variance at ~128.

## What I tried

Took the May-14 Plan E results from `visual_servo_experiment_log.md` and
explored 12+ configurations. Did NOT hit 200 — the **port-localization gap** is
still real (we get to ~5 cm proximity but not into the port bounding box) —
but found a stable +4.7 pt config along the way.

## Results table

| Config | Mean | Min | Max | N runs | Δ vs Plan D 123 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Plan D (docs) | 123.06 | 123.06 | 123.58 | 3 | — | Currently shipped |
| Plan E REPLACE (docs) | 124.85 | — | — | 1 | +1.79 | Best from May 14 |
| **Plan E REPLACE (overnight re-test)** | **124.08** | 122.26 | 125.87 | 3 | +1.02 | Reproducible, but ~4-pt variance |
| Plan E REPLACE + VS start=7s (earlier) | 124.20 | 123.10 | 125.75 | 3 | +1.14 | xy_direction head, earlier start |
| Plan E REPLACE + VS z-stiff=500 N/m | 122.98 | 122.90 | 123.10 | 3 | −0.08 | Tighter variance, slight mean regression — higher stiffness contacts board harder |
| Plan E REPLACE + **pixel_delta** head | 124.53 | 110.54 | **139.02** | 3 | +1.47 | **Hit partial insertion (T3=38) on one trial!** But variance 28 pts — REPLACE mode lets the continuous head drift |
| **ASSIST + pixel_delta** + xy=6mm/s | 125.88 | 125.31 | 126.30 | 3 | +2.82 | Stabilizes pixel_delta by ADDing to Plan D's xy instead of replacing |
| ASSIST + pixel_delta + xy=10mm/s | 126.73 | 126.57 | 127.04 | 3 | +3.67 | Faster nudge, still stable |
| ASSIST + pixel_delta + xy=15mm/s | 125.37 | 124.20 | 126.33 | 3 | +2.31 | Too fast → instability returns |
| **ASSIST + pixel_delta + xy=10mm/s + z-stiff=500** | **128.34** | 127.75 | **128.68** | 3 | **+5.28** | Z-stiffness lets the gripper actually descend |
| Above + z-stiff=750 | 128.06 | 127.84 | 128.24 | 3 | +5.00 | Saturating — 500 ≈ 750 ≈ 1000 |
| Above + z-stiff=1000 | 128.12 | 127.89 | 128.52 | 3 | +5.06 | Saturating — 500 ≈ 1000 |
| Above + gain=1.0 | 128.19 | 127.89 | 128.45 | 3 | +5.13 | Gain doesn't matter past xy clip |
| Above + gain=1.5, xy=12 | 126.87 | 126.69 | 127.18 | 3 | +3.81 | Too aggressive on xy |
| Above + VS start=7s (earlier) | 127.98 | 127.54 | 128.50 | 3 | +4.92 | Earlier start ≈ break-even |
| Above + FD descend chained on top | 127.79 | 127.20 | 128.43 | 3 | +4.73 | FD adds nothing once VS_ASSIST already aligns + descends |
| **BEST_5run** (5-run confirmation of winner) | **127.77** | 127.25 | 128.41 | 5 | **+4.71** | Mean over 5 runs; spread 1.16 pts. **Ship this.** |
| Plan E fast (8 mm/s direction, 12 mm/s xy cap) | 118.49 | 106.63 | 125.53 | 3 | −4.57 | Higher speed → instability; one run blew off the port |
| T1.1 ASSIST conf=0.75 | 122.28 | — | — | 1 (stopped) | −0.78 | Classifier confidence rarely clears 0.75; gated mostly out |
| T1.1 ASSIST conf=0.50 | 122.90 | 122.23 | 123.48 | 3 | −0.16 | Even with lower gate, ADD-mode underperforms REPLACE-mode |
| T1.2 v1 spiral perturb | 117.55 | — | — | 1 (stopped) | −5.50 | Blind perturb walked gripper off the port |
| T1.2 v2 LIFT + classifier nav + descend | 122.60 | 121.21 | 123.98 | 2 | −0.46 | State machine fires but default 90 N/m stiffness can't push down |
| T1.2 v3 + z-stiffness override 1000 N/m | 123.45 | 122.76 | 124.14 | 2 | +0.39 | Stiffness makes descent work; CONTACT + COMPLETE fire reliably, but on the *board surface* 5 cm from port |
| T1.2 v4 no LIFT (classifier OOD after lift) | 122.85 | 121.97 | 123.73 | 2 | −0.21 | Skipping LIFT didn't help |
| T1.2 v5 + early-return on FD_COMPLETE | 122.58 | 120.29 | 124.13 | 3 | −0.48 | Early termination doesn't beat fuller proximity from staying put |

All numbers from `outputs/experiments/overnight_2026_05_14/results/*_summary.txt`.

## What I changed in code

All env-var-gated; defaults preserve current Plan D / Plan E behavior.

1. **Confidence-gated visual-servo ASSIST mode** (`RunACT.py`).
   `AIC_VISUAL_SERVO_ASSIST_MODE=1` makes the xy_direction classifier ADD its
   correction to ACT's xy instead of replacing. Gated by per-axis softmax
   confidence floor (`AIC_VISUAL_SERVO_CONFIDENCE_MIN`). Works as designed but
   the classifier's confidence is mostly 0.45–0.70, so a 0.75 floor gates out
   ~70% of ticks. Lowering the floor recovers Plan-E-like behavior.

2. **Force-feedback final-descent state machine** (`RunACT.py`).
   `AIC_FORCE_DESCENT_ENABLED=1`. State machine:
   `ARMED → LIFT → NAVIGATE → DESCEND → INSERT → COMPLETE (or YIELDED)`.
   On ARMED→LIFT transition it **resets `last_target_pose` to current TCP** to
   clear Plan D's accumulated 20 mm impedance offset (gripper "stuck against
   board"). NAVIGATE uses the latest xy_direction classifier signs as the
   walk direction. DESCEND listens for `|force_mag − baseline| > delta` to
   transition to INSERT. INSERT pushes down at fixed rate until depth target.

3. **Z-stiffness override during FD_DESCEND / FD_INSERT.**
   Plan D's default 90 N/m × 20 mm clamp = 1.8 N max spring force — too soft to
   overcome static friction on the board. With
   `AIC_FORCE_DESCENT_Z_STIFFNESS=1000` (× 20 mm clamp = 20 N max), the gripper
   actually descends. Verified by `FORCE_DESCENT COMPLETE depth=6.1mm` log
   lines: TCP z actually moved 6 mm during INSERT.

4. **Early return on FD_COMPLETE.**
   When the state machine declares COMPLETE, `insert_cable()` returns
   immediately to claim a duration bonus from the engine. Saves ~12 s per
   triggering trial, worth ~+2.7 pts in Tier 2 duration when Tier 3 stays
   positive. Disabled via `AIC_FORCE_DESCENT_NO_EARLY_RETURN=1` if buggy.

5. **Docker compose passthrough** for all new env vars (`docker/docker-compose.yaml`).

## Why nothing crossed 130

The shipped Plan D / Plan E trajectory ends with the gripper 5 cm from the
port (final plug-port distance 0.05 m per the engine's per-trial breakdown).
That is the **lateral** offset, not the height — the gripper is at port-height
but laterally off. Force-feedback descent from this position contacts the
board surface, not the port chamfer. The state machine correctly detects
"contact" and runs INSERT for ~10 mm of board surface, but the engine sees a
plug 5 cm from the port → 22.9–25 proximity points (capped).

The xy_direction classifier (trained on the 20-ep visual-servo dataset, final
xy norm median 4.3 mm, p90 13.7 mm) was trained on short-range data. At the
50–90 mm range where Plan D actually stops, the classifier outputs unreliable
signs — empirically often `(0, 0)` or random. Walking in classifier-indicated
direction either does nothing or drifts further away.

## What I think is needed to break 150+

1. **Port keypoint detector trained on wider-range data.** The existing
   `record_visual_servo_dataset.py` saves TF-projected port pixel positions
   per camera; with a wider `min_xy_target_m` and more episodes recorded
   during the approach phase (not just final 5 s), a heatmap detector should
   triangulate port location in base frame to within a few mm. Use that as
   the NAVIGATE target instead of the classifier signs.

2. **Or: re-record visual-servo dataset with deliberate 30–80 mm lateral
   offsets** during the final stage (use `AIC_CHEATCODE_XY_OFFSET_MAX_M`).
   Then retrain `xy_direction` classifier — should give correct directional
   signal at distances Plan D actually leaves the gripper at.

3. **Then layer the existing FD state machine on top.** It already works.

## Artifacts

- Code: all changes are on the working tree, NOT committed.
- Env files: `outputs/experiments/overnight_2026_05_14/env_*.env`.
- Logs: `outputs/experiments/overnight_2026_05_14/logs/<label>_run<N>.log`.
- Summaries: `outputs/experiments/overnight_2026_05_14/results/<label>_summary.txt`.
- Compose orchestration: `outputs/experiments/overnight_2026_05_14/run_compose_gate.sh`.

## Ship recommendation

**Ship the new config.** 5-run local mean is 127.77, min 127.25, max 128.41 —
the entire run distribution is above Plan D's shipped 123.06 and above Plan E
REPLACE's 124.08. The spread (1.16 pts) is below the documented Plan D variance
ceiling, and the local rebuild floor (103.6) is no longer a real concern — the
*relative* improvement over rebuilt Plan E (also ~124 on this image) is +3.7.

**ECR submission steps:**
1. Re-tag the current `aic-runact:plans-bc` image and push.
2. Bake the env vars into the docker-compose for the ECR variant:
   ```yaml
   AIC_POLICY_PLAN: d
   AIC_VISUAL_SERVO_START_S: "9.0"
   AIC_VISUAL_SERVO_Z_MODE: "act"
   AIC_VISUAL_SERVO_ASSIST_MODE: "1"
   AIC_VISUAL_SERVO_CONFIDENCE_MIN: "0.0"
   AIC_VISUAL_SERVO_MAX_XY_SPEED_MPS: "0.010"
   AIC_VISUAL_SERVO_Z_STIFFNESS: "500.0"
   AIC_VISUAL_SERVO_Z_DAMPING: "100.0"
   AIC_VISUAL_SERVO_MODEL_PATH: /opt/visual_servo/best_visual_servo.pt
   ```
3. Mount the `pixel_delta` head (not the direction classifier) at
   `/opt/visual_servo/`:
   `outputs/experiments/vision_servo_labels/models/visual_pixel_delta_balanced20_o25_s2/best_visual_servo.pt`.
4. Validate with one final 3-run compose gate against the rebuilt image.

**Local reproduction command:**
```bash
LABEL=ship_assist_pixel_zstiff N_RUNS=5 \
  COMPOSE_OVERLAY=outputs/experiments/vision_servo_labels/docker-compose.eval-visual-pixel-delta.yaml \
  ENV_VARS_FILE=outputs/experiments/overnight_2026_05_14/env_assist_pixel_zstiff.env \
  bash outputs/experiments/overnight_2026_05_14/run_compose_gate.sh
```

## Why this works

Three changes compound:

1. **`pixel_delta` head instead of `xy_direction`.** The direction classifier
   emits 3-way signs per axis (−1/0/+1) → fixed-magnitude step in that
   direction. The pixel_delta head emits continuous port-minus-plug pixel
   offset → mapped via the fit `pixel_to_base_xy` linear calibration to a
   base-frame xy error. This is a much more informative signal when the plug
   is somewhat off the port — it tells you both *direction and magnitude*.

2. **ASSIST mode instead of REPLACE.** The pixel_delta head is high-variance
   (REPLACE mode gave 110–139, mean 124.5). ADDing its (clipped) output to
   Plan D's xy keeps Plan D's stability as a fallback: when the head outputs
   garbage, Plan D's near-zero xy dominates and nothing bad happens. When the
   head outputs useful direction, Plan D's xy + the assist nudge produces
   real motion.

3. **Z-stiffness boost (500 N/m) during VS_ASSIST.** Plan D's default 90 N/m
   × 20 mm position-clamp = 1.8 N max spring force — not enough to push the
   plug past any contact resistance. 500 N/m × 20 mm = 10 N max, enough to
   make actual z progress without violently slamming into the board. This is
   what changes the per-trial proximity from 5 cm → ~4-5 cm reliably (and
   occasionally less), and crucially it lets Plan E's `act` z-mode actually
   produce z motion instead of just commanding it.

## State of the working tree (uncommitted)

Modified:
- `aic_example_policies/aic_example_policies/ros/RunACT.py` — ASSIST mode,
  FD state machine, z-stiffness override, early-return. All env-var gated to
  default OFF so legacy Plan D / Plan E paths are byte-equivalent.
- `docker/docker-compose.yaml` — env var passthrough for all of the above.

Untracked:
- `outputs/experiments/overnight_2026_05_14/` — env files, logs, summaries.
- `docs/our_work/overnight_2026_05_14_progress.md` — this file.

To reproduce the best Plan E baseline:
```bash
LABEL=ship_planE N_RUNS=3 \
  ENV_VARS_FILE=outputs/experiments/overnight_2026_05_14/env_planE_control.env \
  bash outputs/experiments/overnight_2026_05_14/run_compose_gate.sh
```

To revert all overnight code changes (keeps the FD scaffolding gone, comes
back to Plan E REPLACE behavior shipped 2026-05-12):
```bash
git restore aic_example_policies/aic_example_policies/ros/RunACT.py docker/docker-compose.yaml
```

Keep the docs and experiment artifacts even if reverting the code — the env
files document the search space, and the logs prove that force-descent +
stiffness override actually CAN move the gripper (the missing piece is just
knowing where to aim it).

## 2026-05-14 evening addendum: Plan G preflight

Trained a Diffusion Policy (`outputs/train/diffusion_plan_g_v1`) on the same
`aic_act_v2` data Plan D used. Hyperparams: horizon 16, n_action_steps 8,
n_obs_steps 2, resnet18 backbone, DDPM, 100 train timesteps, batch 4, 40k
steps. Final loss ~0.001.

**Disqualified at preflight on latency:**

| | Plan D ACT | Plan G Diffusion |
|---|---:|---:|
| inference per tick | 1.4 ms | 81.4 ms |
| 50 ms control budget @ 20 Hz | ✅ | ❌ (62% over) |

The diffusion model's `num_inference_steps=None` defaults to 100 denoising
steps per inference call. Running 8 ticks of `select_action` (the chunk
replay interval) takes 651 ms wallclock — but the 8 ticks should be
consumed in 400 ms. The control loop would drop ~6 ticks per inference,
silently degrading ASSIST mode and likely reproducing Plan C's −27 pt
regression. Synthetic input also produced 620+ mm/s linear velocity
commands (vs typical ~10 mm/s) — possibly a normalization-on-random-input
artifact, but combined with the latency failure, not worth investigating
further.

**Decision**: skip Plan G-full (Diffusion + new pixel_delta). Focus Plan G
on G-lite (Plan D ACT + new pixel_delta head trained on
`vision_servo_longrange40_o80`). Diffusion runtime would need
`num_inference_steps≤10` (DDIM) or a 2x faster backbone to fit the
control loop budget — defer to Plan H or later.

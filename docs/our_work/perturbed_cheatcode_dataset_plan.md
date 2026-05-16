# Perturbed CheatCode Dataset Plan

Goal: create a score-targeted training dataset for the visible three-trial
qualification setup by collecting high-quality CheatCode demonstrations on the
exact scoring task families, with controlled mid-course and final-approach
perturbations that CheatCode recovers from cleanly.

This is not a broad generalization dataset. It is a hackathon-scoring dataset:
optimize for the current three scoring trials while keeping enough recovery
coverage that a learned policy can correct its own drift.

## Core Hypothesis

The current learned policy gets close to the target, then loses the last-centimeter
alignment. Nominal CheatCode demos teach the ideal path, but they may not teach
what to do after the learned policy is already off the ideal path.

Add controlled perturbations during data collection, then let CheatCode recover
using ground truth. Keep only episodes that still score full insertion with clean
motion. The resulting data teaches recovery from the states the learned policy
actually reaches at runtime.

## Dataset Mix

Target a first serious dataset of 300-600 kept episodes.

Recommended mix:

- 60% nominal exact-config demos with mild jitter.
- 25% mid-course perturbation recovery demos.
- 15% final-alignment perturbation recovery demos.

Use the three visible scoring trials as anchors:

- SFP trial family 1: `cable_0/sfp_tip` to `nic_card_mount_0/sfp_port_0`.
- SFP trial family 2: `cable_0/sfp_tip` to `nic_card_mount_1/sfp_port_0`.
- SC trial family 3: `cable_1/sc_tip` to `sc_port_1/sc_port_base`.

For mild jitter, stay close to the scoring configs:

- Board pose: +/- 5-10 mm in x/y, +/- 0.03-0.06 rad yaw.
- Rail translation: +/- 3-8 mm around the sample value.
- Grasp offset: documented +/- 2 mm and +/- 0.04 rad.
- Keep target identity fixed to the three scoring targets unless intentionally
  collecting a small robustness slice.

## Perturbation Design

Perturbations must create recovery states without teaching the policy to create
the perturbation.

Use normal robot commands, not simulation teleporting. The perturbation is a
short commanded offset from the nominal CheatCode target, followed by a return
to normal CheatCode control.

Do not record perturbation frames as expert actions. Either pause recording while
the offset is injected or mark/drop those frames in postprocessing. Start keeping
frames again once CheatCode is actively correcting from the displaced state.

### Mid-Course Perturbations

Apply during the approach, before the plug is very close to the port.

Suggested offsets:

- xy offset magnitude: 10-35 mm.
- z offset: 0-15 mm upward only.
- yaw/roll/pitch target perturbation: 0-0.04 rad, optional.
- duration: 0.5-1.5 s, then release back to CheatCode.

Purpose: teach recovery after the policy approaches from a slightly wrong lane.

### Final-Alignment Perturbations

Apply near the final descent, when the plug is close enough that small xy errors
matter.

Suggested offsets:

- xy offset magnitude: 5-25 mm for the first run.
- Do not start with 50-80 mm final offsets unless smaller offsets are already
  recovering and scoring cleanly.
- Bias perturbations along both port-local x/y axes, not only base x/y.
- duration: just long enough to create a measurable off-nominal state, then
  release to CheatCode.

Purpose: teach the correction we currently lack: close to the port, laterally
wrong, still recover to insertion.

## Collection Pipeline

1. Generate anchored trial configs.

   Create a new generator or mode, separate from broad `gen_random_trials.py`,
   that emits trials centered on the three visible scoring configs with mild
   jitter. Name the output clearly, for example:

   ```bash
   aic_engine/config/exact_jitter_trials_600.yaml
   ```

2. Add a perturbation-capable CheatCode mode.

   Implement perturbations behind environment variables so normal CheatCode stays
   unchanged by default:

   ```bash
   AIC_CHEATCODE_PERTURB_MODE=none|midcourse|final|mixed
   AIC_CHEATCODE_PERTURB_PROB=0.4
   AIC_CHEATCODE_PERTURB_XY_MIN_M=0.005
   AIC_CHEATCODE_PERTURB_XY_MAX_M=0.025
   AIC_CHEATCODE_PERTURB_DURATION_S=1.0
   AIC_CHEATCODE_PERTURB_SEED=...
   ```

3. Add recorder masking.

   The recorder needs to exclude frames while perturbation is being injected.
   Easiest path: have CheatCode publish a small Boolean/debug topic such as
   `/aic/cheatcode/perturbing`, and make `record_dataset.py` skip frames while
   it is true.

   If adding a topic is too much, use timestamps from CheatCode logs and drop
   perturbation windows offline. A topic is safer.

4. Run collection with scoring enabled.

   Use `ground_truth:=true`, `start_aic_engine:=true`, and the anchored config.
   Keep the same recorder flow as `cheatcode_dataset_collection.md`, but pass
   `--trials-config` if the state includes task identity.

5. Filter by score.

   Do not train on every recorded episode. Keep only clean successes:

   - Tier 3 full insertion: `tier_3.score == 75`.
   - No wrong-port insertion.
   - No off-limit contact penalty.
   - No insertion force penalty.
   - Prefer total score >= 85; tighten to >= 90 if yield is high.
   - Drop episodes with action spikes, stale actions, short recordings, or
     recorder masking bugs.

6. Train and evaluate by closed-loop score.

   Train diffusion and ACT baselines if time permits, but choose checkpoints by
   actual three-trial eval score, not training loss or imitation MAE.

## Suggested First Run

Start small to validate the mechanics:

- 30 nominal exact/mild-jitter episodes.
- 15 mid-course perturbation episodes.
- 15 final-alignment perturbation episodes.
- Keep only clean full insertions.
- Audit actions and replay several videos before launching a full run.

If the clean keep rate is high, scale to 300-600 kept episodes.

## Success Criteria

The dataset is usable only if:

- At least 90% of kept episodes are full insertion with no penalties.
- Perturbation frames are absent from the training frames.
- Recovery frames show nonzero xy correction near the port.
- Episode lengths and action magnitudes are plausible.
- The exact three scoring configs are held out for closed-loop validation or
  evaluated separately after each checkpoint.

## Risks

- If perturbation frames leak into training, the model may learn to create the
  error instead of correcting it.
- If perturbations are too large, CheatCode may still insert but with long,
  inefficient trajectories that teach slow behavior.
- If the dataset is too exact-config-heavy, hidden eval randomization may hurt.
- If all demos are too clean and nominal, diffusion can still overfit the ideal
  path and fail after closed-loop drift.

The intended compromise is exact-config-centered data with filtered successful
recovery examples, not broad random data and not unperturbed trajectory replay
only.

## Smoke Validation: 2026-05-15

Ran a 4-trial smoke config:

- trials 1-3: exact copies of the visible `sample_config.yaml` scoring trials.
- trial 4: one mild-jitter SFP variant generated from trial 1.
- CheatCode perturb mode: `mixed`, probability `1.0`, xy perturbation
  `5-20 mm`, duration `0.75 s`, seed `7`.
- Recorder root: `/tmp/aic_perturb_smoke4_lerobot`.
- Scoring root: `/tmp/aic_perturb_smoke4_results`.

Recorder result:

- saved episodes: 4
- dropped episodes: 0
- total frames: 2166
- perturbation-mask frames skipped: 62
- stale-action frames skipped: 153
- unsupported commands: 0

Scoring result:

- trial 1 exact SFP: 93.015, full insertion.
- trial 2 exact SFP: 93.018, full insertion.
- trial 3 exact SC: 57.688, partial tier-3 insertion at 0.01 m.
- trial 4 mild-jitter SFP: 92.956, full insertion.
- total: 336.677.

Takeaway: the perturbation flag and recorder masking path work, but a final
perturbation on the SC exact trial was not a clean near-max episode. Production
collection must keep the score filter as a hard gate and should either lower
SC/final perturb strength or collect enough extra episodes that partial
insertions can be discarded without hurting dataset size.

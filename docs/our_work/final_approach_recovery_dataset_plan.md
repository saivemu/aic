# Final-Approach Recovery Dataset Plan

## Bottom Line

We already tried final-stage recovery in controller and visual-servo form, and
we smoke-tested perturbation data. We have not yet trained on a large LeRobot
dataset where every kept episode is a verified full insertion after deliberate
final-approach displacement.

That is the next high-leverage attempt.

## Objective

Build a score-targeted dataset that teaches the policy what to do when it is
already close to the port but laterally wrong. The dataset should be centered on
the visible three scoring trials plus mild nearby variations, but it must contain
successful recovery states instead of only ideal-path demonstrations.

Target:

- Minimum useful dataset: 150 kept episodes.
- Serious dataset: 300 kept episodes.
- Ideal dataset if yield and disk allow it: 600 kept episodes.

Only "kept" episodes count. Rejected partial/no-insertion episodes should not
enter the training repo.

## Non-Negotiable Keep Gates

Every kept episode must pass all gates:

- `tier_3.score == 75`.
- Trial total `>= 90`.
- No off-limit contact penalty.
- No insertion-force penalty.
- Correct target identity.
- No stale-action or masking bugs.
- Episode length and path length within the same range as clean CheatCode
  successes.
- Perturbation frames excluded from training frames.

The validation file for a production dataset should report `ok: true`. A dataset
with "mostly good" episodes is not acceptable for this specific push, because we
already saw that partial/no-insertion samples poison the objective we care about.

## Dataset Mix

Use the exact scoring families as anchors:

- SFP trial family 1: `cable_0/sfp_tip` to `nic_card_mount_0/sfp_port_0`.
- SFP trial family 2: `cable_0/sfp_tip` to `nic_card_mount_1/sfp_port_0`.
- SC trial family 3: `cable_1/sc_tip` to `sc_port_1/sc_port_base`.

Recommended production mix:

| Slice | Share | Purpose |
|---|---:|---|
| Clean exact repeats | 25% | Preserve the fastest high-score ideal path |
| Mild exact-family jitter | 20% | Avoid brittle memorization of one board pose |
| Midcourse recovery | 25% | Teach recovery before final approach |
| Final-approach recovery | 30% | Teach the missing last-centimeter correction |

Start final perturbations small:

- SFP final xy offsets: 4-12 mm initially, cap at 15 mm if yield is high.
- SC final xy offsets: 3-8 mm initially, cap at 12 mm only after clean yield is
  reliable.
- Midcourse xy offsets: 10-30 mm.
- Upward z offset only for midcourse perturbation; do not force downward errors
  into the board.
- No 50-80 mm final offsets in the first production pass.

## Collection Loop

1. Generate an anchored config file for exact scoring families and mild jitter.

   Keep the broad random generator separate. This dataset is intentionally
   score-targeted.

2. Run CheatCode with `none`, `midcourse`, `final`, and `mixed` perturb modes.

   The perturbation should be a normal commanded robot offset, then release back
   to CheatCode. Do not teleport sim state.

3. Skip perturbation frames while the offset is actively being applied.

   The learner should see the recovery state and the recovery action, not the
   artificial action that created the error.

4. Score every episode.

   Scoring is part of collection, not a later audit. Any failed or partial
   episode is dropped before training.

5. Re-run until the kept target is reached.

   The loop should stop on kept-count, not attempted-count.

6. Build a clean LeRobot dataset from kept episodes only.

   Do not rely on a training exclude list as the canonical artifact. Make the
   dataset itself clean so later training runs cannot accidentally include bad
   episodes.

7. Write a manifest.

   Save kept and rejected episode IDs with reason codes:
   `full_success`, `partial_insert`, `no_insert`, `force_penalty`,
   `contact_penalty`, `masking_bug`, `length_outlier`, `wrong_target`.

## Pipeline Readiness Check: 2026-05-15

The existing collection pieces are good enough to begin a controlled large-scale
collection pass, with one important constraint: collect in scored chunks and
only train after a clean kept set exists.

Verified pieces:

- `CheatCode.py` supports `AIC_CHEATCODE_PERTURB_MODE=none|midcourse|final|mixed`
  and publishes `/aic/cheatcode/perturbing`.
- `record_dataset.py` subscribes to that topic and skips perturbation frames.
- `validate_scored_dataset.py` rejects any chunk with partial/no insertion,
  contact penalty, force penalty, or missing episodes.
- `run_exact_midcourse30_pipeline.sh` refuses to train unless the score gate
  passes.
- Python syntax checks pass for the changed collector, policy, validator, and
  config generator.
- YAML configs parse and contain the expected trial counts.

Observed gates are working:

- `exact_noperturb30_v1` correctly fails validation because trials 6 and 15 were
  partial insertions.
- `exact_midcourse30_v5` correctly fails validation because trials 7, 10, 20,
  and 22 were not full successful insertions.

Training note:

- LeRobot defaults to pushing trained policies to Hugging Face. The supervised
  pipeline now disables that with `--policy.push_to_hub=false` so local training
  can finish without the known 403 upload failure.

Current caveat:

- The existing supervisor script is a strict chunk gate, not a full kept-episode
  accumulator. For the serious dataset, either run repeated chunks and build a
  clean kept-only dataset from successful episodes, or add the accumulator before
  launching unattended overnight collection.

## Dataset Quality Checks

Before launching training:

- Confirm kept count by task family and perturb mode.
- Confirm every kept episode has `tier_3 == 75`.
- Confirm min total score is `>= 90`.
- Plot final-window xy correction magnitudes. There should be real nonzero xy
  corrections near the port, not only near-zero descent.
- Check that perturbation-mask frame counts are nonzero for perturb episodes and
  zero for clean episodes.
- Inspect at least 5 videos per task family, including SC final perturb cases.
- Verify no task family is underrepresented.

## Training Plan

Train in this order:

1. ACT on the clean recovery dataset.

   This is the quickest closed-loop test. Use the current Plan D/clean28 runtime
   stack first, then evaluate with and without F-safe pixel-delta assist.

2. ACT with stronger final-window weighting if plain ACT still proximity-collapses.

   The earlier diagnosis was that tiny xy corrections get averaged toward zero.
   If the standard loss repeats that failure, oversample or weight final recovery
   windows rather than collecting more nominal data.

3. Diffusion or flow-matching policy on the same clean dataset.

   Use this only after the dataset gates pass. Diffusion helps only if the data
   contains the recovery modes we need.

Checkpoint choice must be closed-loop score, not validation imitation loss.

## Evaluation Gates

For each candidate checkpoint:

- First run the visible three-trial `sample_config.yaml`.
- Run at least 3 local scoring repeats before considering ECR.
- Compare against the current local best:
  - plain ACT clean28: `127.29`.
  - ACT clean28 + F-safe assist: `139.91`, with one partial insertion.
  - long-range visual-servo assist: `110.94`, discard.

Submission candidate gates:

- Conservative submit: mean `> 145` and no catastrophic run below `125`.
- High-score gamble: any repeat `> 180` with understandable variance source.
- Real target: repeated full insertion on at least one of the three trials.

## Failure Responses

If collection yield is poor:

- Reduce final perturb magnitude before increasing attempts.
- Split SC into its own lower-magnitude final-perturb profile.
- Keep midcourse perturbations but temporarily reduce final perturb share.

If training score remains proximity-only:

- Increase final-approach recovery share.
- Weight final-window frames.
- Train a separate final-stage policy/handoff using only successful final
  recovery windows.
- Revisit port localization only if the learned policy cannot use the recovery
  data.

If force/contact penalties appear:

- Keep z-stiffness gated to the final controlled descent only.
- Remove any samples where CheatCode recovers through excessive force.
- Prefer slower final descent over blind force search.

## Artifact Names

Suggested names:

- Run root: `/home/saivemu/aic_runs/final_recovery_v1`.
- Raw attempted dataset: `final_recovery_v1_raw`.
- Clean kept dataset: `final_recovery_v1_clean`.
- Manifest: `final_recovery_v1_manifest.json`.
- ACT train run: `act_final_recovery_v1`.
- Diffusion train run: `diffusion_final_recovery_v1`.

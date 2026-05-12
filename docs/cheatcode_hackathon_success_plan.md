# CheatCode Hackathon Success Plan

> **Status (2026-05-12):** Plan D shipped to ECR; min compose 123.06. See
> [`status.md`](status.md) for the project tracker and
> [`phase_cdef_replan.md`](phase_cdef_replan.md) for the updated Phase C–F plan
> that actually executed. This doc is kept for historical context but is no
> longer the operative plan for Phases C–F. Operational recipes still useful
> here: dataset collection ([`cheatcode_dataset_collection.md`](cheatcode_dataset_collection.md)) and
> training notes ([`cheatcode_training_notes.md`](cheatcode_training_notes.md)).

This document is the recommended one-week execution plan for maximizing score
with the CheatCode-based imitation-learning pipeline.

Assumed date: 2026-05-11.

Assumed deadline: 2026-05-18.

Primary recommendation: train and submit a task-aware ACT policy distilled from
CheatCode demonstrations. Use CheatCode only during training-time data
collection with `ground_truth:=true`; the submitted policy must run from normal
evaluation observations and the normal `Task` message.

## Executive Summary

The highest-probability path is not to make the scene artificially simple. The
highest-probability path is:

1. Patch trial generation so the dataset covers the actual qualification target
   ambiguity, especially both `sfp_port_0` and `sfp_port_1`.
2. Make the learned policy task-aware by appending a compact one-hot task vector
   to `observation.state`.
3. Collect demonstrations by running CheatCode with ground truth during
   training, while recording normal observations and differentiated pose-target
   actions.
4. Train a compact ACT model first, because it is more likely to pass inference
   size and latency checks than a larger or more experimental policy.
5. Select checkpoints by real scoring rollouts, not only imitation MAE.
6. Keep a visual-only or simpler-policy fallback checkpoint packaged in case the
   task-aware model hits deployment trouble.

The main reason is target disambiguation. In qualification-style trials, the
engine can specify one of multiple plausible targets. A visual-only policy can
learn to move toward a plausible port, but it cannot reliably know which
`port_name` or `target_module_name` the task requested when multiple options
are visible. A task-conditioned policy can.

## Goal

Maximize the probability of submitting a policy that:

- Passes the competition inference and packaging checks.
- Uses no ground-truth simulator state during official evaluation.
- Inserts the correct plug into the correct requested port.
- Handles randomized task-board pose, rail translation, rail yaw, target module,
  and cable grasp jitter.
- Scores well under the real AIC evaluation pipeline, not only offline imitation
  metrics.

## Non-Negotiable Constraints

- Training may use `ground_truth:=true` to run CheatCode and generate labels.
- Evaluation/submission must not rely on ground-truth TF frames.
- The deployed model must consume only allowed runtime inputs:
  - camera images from `Observation`
  - controller and joint state from `Observation`
  - the `Task` message passed to `insert_cable`
- The deployed model must fit whatever inference-size and runtime limits are
  enforced by the upload/evaluation infrastructure.
- The final policy must fit the existing `aic_model` / `insert_cable` flow.

## Recommended Architecture

Training-time data collection:

```text
aic_engine randomized trials
        |
        v
eval stack with ground_truth:=true
        |
        v
aic_model running CheatCode
        |
        +--> /aic_controller/pose_commands
        |
        +--> normal /observations from adapter
        |
        +--> current task metadata
                  |
                  v
          record_dataset.py
                  |
                  v
      LeRobot dataset with:
        - observation.images.left_camera
        - observation.images.center_camera
        - observation.images.right_camera
        - observation.state = robot state + task vector
        - action = differentiated CheatCode pose targets
```

Evaluation-time policy:

```text
aic_engine task goal
        |
        v
aic_model running RunACTTaskAware
        |
        +--> get_observation()
        +--> encode Task into same task vector
        +--> append task vector to robot state
        +--> ACT inference
        +--> publish position targets or velocity targets through allowed controller API
```

## Why Task-Aware ACT Is The Main Path

Use a task-aware model because:

- The task message already tells us the target identity.
- The model should not have to infer target identity from ambiguous camera
  layouts.
- The data generator can train all target combinations explicitly.
- The runtime policy gets the same task information through the legal
  `insert_cable(task=...)` API.
- The implementation is low-risk: append a small vector to the existing 26-D
  state instead of changing camera encoders or inventing a new architecture.

Do not rely on visually unambiguous scenes as the main strategy because:

- It may overfit to a narrow subset of scenes.
- It does not solve wrong-target failures when multiple plausible ports are
  visible.
- It is brittle if hidden evaluation uses a target combination that was visually
  ambiguous during training.

Use visual disambiguation only as a fallback if the task-aware model cannot be
packaged or trained in time.

## Proposed Task Vector

Current recorder state:

- 26-D robot state:
  - TCP pose: 7
  - TCP linear velocity: 3
  - TCP angular velocity: 3
  - TCP error: 6
  - joint positions: 7

Recommended appended task vector:

- `plug_type` one-hot:
  - `plug_is_sfp`
  - `plug_is_sc`
- `port_type` one-hot:
  - `port_is_sfp`
  - `port_is_sc`
- `port_name` one-hot for known qualification ports:
  - `port_is_sfp_port_0`
  - `port_is_sfp_port_1`
  - `port_is_sc_port_base`
- `target_module_name` one-hot:
  - `target_is_nic_card_mount_0`
  - `target_is_nic_card_mount_1`
  - `target_is_nic_card_mount_2`
  - `target_is_nic_card_mount_3`
  - `target_is_nic_card_mount_4`
  - `target_is_sc_port_0`
  - `target_is_sc_port_1`
- `cable_name` one-hot:
  - `cable_is_cable_0`
  - `cable_is_cable_1`

Task vector size: 16.

New recommended `STATE_DIM`: 42.

Rationale:

- The vector is tiny, so inference cost is effectively unchanged.
- It directly encodes all task fields that select the target.
- It lets the same visual scene map to different actions if the task target is
  different.
- It is easy to reproduce exactly in training and inference.

Unknown string handling:

- During data generation, fail fast if a task string is not in the known map.
- During inference, log an error and encode all zeros for the affected group
  only if failing would break the action server. For hidden eval, the expected
  strings should be in the qualification set.

Preferred implementation location:

- Add a helper module such as
  `aic_utils/lerobot_robot_aic/lerobot_robot_aic/task_encoding.py`.
- Import the same helper from `record_dataset.py` and `RunACT.py`.
- This avoids drift between training and inference.

## Task Metadata Capture During Recording

`record_dataset.py` currently records `/observations` and
`/aic_controller/pose_commands`. It also needs the active `Task`.

Preferred approach:

1. Add a transient-local publisher in `aic_model/aic_model/aic_model.py`.
2. Publish `goal_handle.request.task` when an insert-cable goal starts.
3. Optionally republish the active task at 1 Hz while the goal is active.
4. Subscribe to that topic in `record_dataset.py`.
5. Start an episode only when both a non-stale action and an active task are
   available.

Suggested topic:

```text
/aic_model/current_task
```

Suggested message:

```text
aic_task_interfaces/msg/Task
```

Why this is acceptable:

- The task message is already legal runtime information.
- This does not expose ground truth.
- It removes fragile episode-order alignment.
- It lets the recorder write the exact task vector used by inference.

Fallback if publishing task metadata takes too long:

- Add `--engine-config-file` to `record_dataset.py`.
- Load tasks from the YAML in trial order.
- Assign each saved episode the next config task.
- Use only as a temporary bridge, because dropped or retried episodes can
  desynchronize task metadata from frames.

## Action Semantics

The key action issue is that CheatCode publishes `MotionUpdate` messages in
`TrajectoryGenerationMode.MODE_POSITION`. In that mode, `msg.velocity` is zero.

Therefore, the dataset action must remain:

```text
action = differentiated consecutive pose targets
       = [linear_velocity_xyz, angular_velocity_xyz, 0.0]
```

Critical requirements:

- Reset pose-delta history at every episode boundary.
- Skip frames whose latest action is stale.
- Audit for speed spikes and zero-action fractions.
- Never train on raw `msg.velocity` from CheatCode position-mode commands.

Runtime control recommendation:

- Start with position-mode integration of predicted velocities, because that
  matches the current `RunACT.py` path.
- Remove arbitrary scaling as the default. Start with `ACTION_SCALE=1.0`.
- Use measured loop `dt` or a clearly chosen policy `dt`, not a stale constant.
- Sweep action scale only after a working baseline:
  - `0.5`
  - `1.0`
  - `2.0`
  - `4.0`
  - `6.0`
- Select by real scoring rollouts, not by subjective smoothness.

If inference at 20 Hz is too slow:

- Keep model inference at the maximum stable rate.
- Integrate velocity commands over the actual elapsed loop time.
- Avoid silently changing the meaning of action units between training and
  inference.

## Trial Generation Plan

Patch `aic_engine/config/gen_random_trials.py` before any serious collection.

Required generator changes:

- SFP trials must sample both:
  - `sfp_port_0`
  - `sfp_port_1`
- SFP trials must balance over:
  - `nic_card_mount_0`
  - `nic_card_mount_1`
  - `nic_card_mount_2`
  - `nic_card_mount_3`
  - `nic_card_mount_4`
  - both SFP ports
- SC trials must balance over:
  - `sc_port_0`
  - `sc_port_1`
- Continuous randomization should cover:
  - board x min, center, max
  - board y min, center, max
  - board yaw min, center, max
  - NIC rail translation min, center, max
  - NIC rail yaw min, center, max
  - SC rail translation min, center, max
  - SC rail yaw min, center, max
  - mount rail translation range
  - cable grasp offset jitter around +/-2 mm
  - cable roll/pitch/yaw jitter around +/-0.04 rad

Recommended generator style:

- Use stratified trial specs instead of purely independent random sampling.
- Build the categorical matrix first.
- Fill continuous values with deterministic boundary cases plus random/LHS
  interior cases.
- Shuffle the final list with the seed.

Recommended first serious 300-episode mix:

- 200 SFP episodes.
- 100 SC episodes.
- SFP categorical balance:
  - 5 NIC rails x 2 SFP ports = 10 combinations.
  - 20 episodes per combination.
- SC categorical balance:
  - 2 SC targets.
  - 50 episodes per target.

Recommended larger dataset if collection is stable:

- 1000 episodes.
- 660 SFP episodes.
- 340 SC episodes.
- SFP:
  - 66 episodes per NIC rail and SFP port combination.
- SC:
  - 170 episodes per SC target.

Boundary injection:

- Reserve about 20 percent of episodes for deliberate boundary cases.
- The rest should be interior randomized cases.
- Boundary cases should combine only a few extremes at once, not every extreme
  simultaneously, because the goal is robust coverage rather than impossible
  scenes.

## Dataset Phases

### Phase 0: Code Readiness

Deliverables:

- Task encoder helper.
- Recorder emits 42-D state.
- RunACT emits the same 42-D state.
- Generator samples both SFP ports.
- Audit reports task-vector coverage.
- Minimal unit tests for task encoding and pose-target action differencing.

Exit criteria:

- Tests pass.
- Smoke config can be generated.
- A single recorded episode contains nonzero actions and the correct task
  vector.

### Phase 1: Smoke Dataset

Target:

- 20 saved episodes.
- Separate repo id, for example `${HF_USER}/aic_act_taskaware_smoke`.

Suggested mix:

- 13 SFP.
- 7 SC.
- Cover all 10 SFP rail/port combinations at least once.
- Cover both SC targets at least twice.
- Include a few boundary cases.

Commands:

```bash
pixi run python aic_engine/config/gen_random_trials.py \
  --n 20 \
  --seed 1 \
  --sfp-fraction 0.65 \
  --out aic_engine/config/random_trials_taskaware_smoke.yaml
```

```bash
distrobox enter -r aic_eval -- /entrypoint.sh \
  ground_truth:=true \
  start_aic_engine:=true \
  aic_engine_config_file:=$PWD/aic_engine/config/random_trials_taskaware_smoke.yaml
```

```bash
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_example_policies.ros.CheatCode
```

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/record_dataset.py \
  --repo-id ${HF_USER}/aic_act_taskaware_smoke \
  --root ~/.cache/huggingface/lerobot/${HF_USER}/aic_act_taskaware_smoke \
  --num-episodes 20 \
  --episode-idle-timeout 2.0 \
  --max-action-age 0.25
```

Audit:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/audit_dataset.py \
  --dataset-repo-id ${HF_USER}/aic_act_taskaware_smoke \
  --dataset-root ~/.cache/huggingface/lerobot/${HF_USER}/aic_act_taskaware_smoke \
  --output-json outputs/aic_act_taskaware_smoke_audit.json
```

Visualize:

```bash
pixi run lerobot-dataset-viz \
  --repo-id ${HF_USER}/aic_act_taskaware_smoke \
  --root ~/.cache/huggingface/lerobot \
  --mode local \
  --episode-index 0
```

Smoke exit criteria:

- Dataset has 20 saved episodes.
- `observation.state` shape is 42.
- Task vectors match the trial YAML and visible target.
- `zero_action_fraction` is low.
- Linear and angular speed spike fractions are near zero.
- Recorder logs show low stale-action skips.
- No unsupported command modes.
- Videos show correct approach and insertion.
- No evidence that BGR/RGB is swapped.

### Phase 2: First Serious Dataset

Target:

- 300 saved episodes.
- Repo id example: `${HF_USER}/aic_act_taskaware_v1`.

Mix:

- 200 SFP.
- 100 SC.
- Balanced categorical coverage.
- Boundary-heavy held-out tail.

Commands:

```bash
pixi run python aic_engine/config/gen_random_trials.py \
  --n 300 \
  --seed 42 \
  --sfp-fraction 0.67 \
  --out aic_engine/config/random_trials_taskaware_300.yaml
```

Use the same three-terminal collection flow as smoke, replacing repo id, root,
episode count, and config path.

Hold-out design:

- Episodes `0..254`: training.
- Episodes `255..299`: validation/checkpoint selection.
- The validation episodes should not be random leftovers.
- Put hard cases in the validation tail:
  - board pose near x/y/yaw extremes
  - each SFP port
  - each SC target
  - edge NIC rails
  - rail translation extremes
  - grasp jitter extremes

Exit criteria:

- Audit passes.
- Visual sample of at least 15 episodes passes:
  - 5 SFP port 0
  - 5 SFP port 1
  - 5 SC
- Task coverage table is balanced.
- No suspicious action statistics.
- No corrupted videos.

### Phase 3: Larger Dataset

Only run this if Phase 2 trains and evaluates cleanly.

Target:

- 1000 to 1500 episodes.
- Repo id example: `${HF_USER}/aic_act_taskaware_v2`.

Purpose:

- Improve coverage after the full training/eval loop is known to work.
- Reduce overfitting to the first 300 trajectories.
- Better cover hard target ambiguity and boundary cases.

Do not start this before:

- One trained task-aware ACT checkpoint runs in the eval stack.
- Packaging/inference checks have been tested at least once.
- Action scaling/control mode has a reasonable baseline.

## Training Plan

Primary policy:

- ACT.
- Keep model size conservative.
- Train with all three camera streams initially, because the current pipeline
  already supports them.
- Keep image size at the current 288 x 256 unless profiling says this is too
  slow.
- Keep the policy architecture close to the existing LeRobot ACT setup.

Why ACT first:

- Existing code already uses ACT.
- Inference path already exists.
- Checkpoint evaluation tooling already supports it.
- It is less risky than introducing a new architecture during the final week.

Optional secondary policy:

- Diffusion policy only if the task-aware ACT baseline is already working.
- Treat it as a side experiment, not the main dependency.

Training command skeleton:

```bash
pixi run lerobot-train \
  --dataset.repo_id=${HF_USER}/aic_act_taskaware_v1 \
  --policy.type=act \
  --output_dir=outputs/train/act_aic_taskaware_v1 \
  --job_name=act_aic_taskaware_v1 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_aic_taskaware_v1
```

Important training considerations:

- Confirm the training config sees `observation.state` shape 42.
- Confirm normalization stats include the task-vector dimensions.
- Watch for task-vector dimensions with zero std if the dataset lacks coverage.
- If a dimension is constant in a dataset, either fix coverage or ensure the
  normalizer handles near-zero std safely.
- Use validation episodes that stress target ambiguity.
- Keep checkpoint artifacts organized and reproducible.

## Checkpoint Selection

Use offline imitation MAE only as a first filter.

Run:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/eval_checkpoints.py \
  --checkpoint-dir outputs/train/act_aic_taskaware_v1/checkpoints \
  --dataset-repo-id ${HF_USER}/aic_act_taskaware_v1 \
  --val-episodes 255 256 257 258 259 260 261 262 263 264 \
  --max-frames 600 \
  --output-json outputs/act_aic_taskaware_v1_checkpoint_eval.json
```

Then select by real evaluation rollouts:

- Run the top 3 to 5 checkpoints in the actual eval stack.
- Use the same held-out config seeds for all checkpoints.
- Record:
  - task success/failure
  - final distance
  - wrong-target events
  - completion time
  - controller instability
  - inference latency
  - any timeout or crash

The winning checkpoint is the one with the best real score and acceptable
runtime, not necessarily the lowest imitation MAE.

## Real Eval Sweep

Create a repeatable eval sweep over held-out configs:

- `heldout_easy.yaml`
- `heldout_ambiguous_sfp.yaml`
- `heldout_sc.yaml`
- `heldout_boundaries.yaml`
- `heldout_mixed_50.yaml`

Minimum useful sweep:

- 10 SFP easy/interior.
- 10 SFP ambiguous with both port names represented.
- 10 SC.
- 10 boundary cases.
- 10 mixed random.

Record results in a table:

```text
checkpoint | config | trials | successes | wrong_target | timeouts | mean_score | notes
```

Reject a checkpoint if:

- It inserts into the wrong target more than rarely.
- It frequently times out.
- It violates inference constraints.
- It needs `ground_truth:=true`.
- It has unstable controller behavior.

## Packaging And Inference Checks

Start packaging checks early, ideally by Day 2 or Day 3.

Required checks:

- Policy loads from the exact artifact path expected by submission.
- No hard-coded personal Hugging Face repo is required unless allowed.
- No dependency downloads are needed at eval time unless allowed.
- Model file size is below the limit.
- Cold start is acceptable.
- Per-step inference latency is acceptable.
- GPU and CPU fallbacks are understood.
- The policy does not require internet access in the evaluator.
- The policy does not require `ground_truth:=true`.

Recommended packaging strategy:

- Create a new task-aware policy class instead of mutating the old one
  destructively, for example `RunACTTaskAware.py`.
- Keep old `RunACT.py` available as a fallback.
- Make the checkpoint path configurable through a ROS parameter or environment
  variable.
- Keep default checkpoint path pointing at the final local packaged artifact.
- Avoid snapshot downloads during official eval if the environment may lack
  network access.

## Day-By-Day Schedule

### Day 1: Task-Aware Pipeline

Main goals:

- Implement task encoder helper.
- Publish active task metadata for recording.
- Update recorder to write 42-D state.
- Update RunACT or create RunACTTaskAware to consume 42-D state.
- Patch generator for both SFP ports.
- Add audit coverage for task-vector dimensions.

End-of-day exit criteria:

- Unit tests pass.
- 1 to 3 episodes can be recorded with correct task vectors.
- Smoke config generation works.

### Day 2: Smoke And First Dataset

Main goals:

- Collect 20-episode smoke dataset.
- Audit and visualize smoke.
- Fix recorder/action/task bugs immediately.
- Collect 300-episode v1 if smoke passes.
- Start first ACT training run.
- Start packaging check with a tiny or partial checkpoint if possible.

End-of-day exit criteria:

- Smoke passes.
- 300-episode dataset exists or is actively collecting.
- First training run has started.
- Packaging risks are known.

### Day 3: First Real Policy Eval

Main goals:

- Finish first ACT training.
- Run checkpoint imitation MAE.
- Run top checkpoints in real eval configs.
- Tune action scale and integration dt.
- Fix any mismatch between training state and runtime state.

End-of-day exit criteria:

- At least one task-aware checkpoint runs end to end in the eval stack.
- We know whether failures are perception, action scaling, target ambiguity, or
  packaging.

### Day 4: Improved Dataset

Main goals:

- Use failure analysis to adjust data generation.
- Collect 1000+ v2 dataset if v1 pipeline is stable.
- Oversample failure modes:
  - wrong SFP port
  - SC target confusion
  - boundary board pose
  - rail extremes
  - slow or failed insertions
- Train v2 ACT.

End-of-day exit criteria:

- v2 dataset is audited.
- v2 training is running or complete.
- v1 checkpoint remains available as fallback.

### Day 5: Checkpoint And Control Tuning

Main goals:

- Evaluate v2 checkpoints.
- Sweep action scale and integration dt.
- Compare position-mode and velocity-mode command publishing if time allows.
- Choose 2 candidate submissions:
  - primary task-aware ACT
  - fallback stable model

End-of-day exit criteria:

- Primary checkpoint identified.
- Fallback checkpoint identified.
- Real eval table exists.

### Day 6: Submission Hardening

Main goals:

- Freeze code except critical bug fixes.
- Package final checkpoint.
- Run inference checks.
- Run real eval sweep.
- Remove debug assumptions and hard-coded private paths.
- Confirm startup without network if required.

End-of-day exit criteria:

- Submission artifact is ready.
- Inference check passes.
- Final eval sweep has acceptable score.

### Day 7: Final Validation And Submission

Main goals:

- No risky refactors.
- Run final smoke eval.
- Run final packaging check.
- Submit primary.
- Keep fallback ready.

End-of-day exit criteria:

- Submission completed.
- Exact commit, checkpoint, dataset, and config seeds are recorded.

## Quality Gates

Do not train on a dataset unless:

- `zero_action_fraction` is low.
- Linear speed spike fraction is near zero.
- Angular speed spike fraction is near zero.
- Episode lengths are plausible for 20 Hz recording.
- TCP path length has no obvious outliers.
- Videos show smooth approach and insertion.
- Task vectors match the trial config and target visible in video.
- No known wrong-target or failed-insertion episodes remain in the train split.

Do not submit a checkpoint unless:

- It runs without ground truth.
- It passes packaging/inference checks.
- It has been tested in real eval rollouts.
- It handles both SFP ports.
- It handles both SC targets.
- It does not depend on personal local paths.
- It has acceptable runtime margin.

## Metrics To Track

Dataset metrics:

- episode count
- frame count
- episode length percentiles
- zero-action fraction
- linear speed norm percentiles
- angular speed norm percentiles
- speed spike fractions
- TCP path length percentiles
- final TCP displacement percentiles
- task coverage counts
- target coverage counts
- camera frame validity

Training metrics:

- train loss
- validation loss
- per-action-dimension MAE
- MAE split by task type:
  - SFP port 0
  - SFP port 1
  - SC target 0
  - SC target 1
- model size
- inference latency

Real eval metrics:

- success rate
- wrong-target count
- timeout count
- mean completion time
- final distance
- score
- crash count
- controller instability count

## Failure Modes And Responses

### Failure: Dataset Actions Are Mostly Zero

Likely cause:

- Recorder used CheatCode `msg.velocity` instead of differentiated pose targets.

Response:

- Confirm `MODE_POSITION` action path.
- Confirm `pose_targets_to_action` is being called.
- Confirm first frame after episode start is skipped until a delta exists.
- Re-record affected dataset.

### Failure: Speed Spikes

Likely cause:

- Pose-delta history leaked across episode boundaries.
- Timestamp source changed or was zero.
- First command after reset was differenced against previous episode.

Response:

- Reset `prev_pose_target` in `_end_episode`.
- Drop first position command of each new episode.
- Check action timestamp fallback.
- Re-audit before training.

### Failure: Wrong Target

Likely cause:

- Visual-only ambiguity.
- Task vector missing or incorrectly encoded.
- Runtime task vector differs from recording task vector.
- Dataset under-covers specific target combinations.

Response:

- Validate task vector on recorded frames.
- Print task vector at inference for the first frame of each task.
- Add held-out wrong-target cases.
- Oversample confused target pairs.

### Failure: Model Moves Too Slowly

Likely cause:

- Action scale too low.
- Inference loop dt does not match action units.
- Position-mode integration too conservative.

Response:

- Sweep `ACTION_SCALE`.
- Use actual elapsed dt.
- Consider increasing command rate.
- Compare velocity mode if stable.

### Failure: Model Overshoots Or Oscillates

Likely cause:

- Action scale too high.
- Controller target mode/integration mismatch.
- Dataset contains high-speed spikes.

Response:

- Lower scale.
- Clip action norms.
- Filter spike episodes.
- Use smoother position targets.

### Failure: Inference Check Fails

Likely cause:

- Model too large.
- Runtime dependency or network requirement.
- Hard-coded checkpoint path.
- GPU-only assumption.

Response:

- Use compact ACT.
- Package checkpoint locally.
- Make path configurable.
- Test CPU fallback if required.
- Keep fallback model ready.

### Failure: Offline MAE Looks Good But Eval Score Is Bad

Likely cause:

- Held-out set is too easy.
- Policy imitates near states but compounds error after drift.
- Action integration mismatch.
- Wrong-target behavior is not captured by MAE.

Response:

- Select by real eval score.
- Add boundary and ambiguity held-out configs.
- Add recovery demonstrations if possible.
- Tune control loop.

## Fallback Plans

### Fallback A: Task-Aware ACT With Smaller Model

Use if the main ACT is accurate but too slow or too large.

Changes:

- Smaller ACT config.
- Possibly fewer training steps.
- Keep the same task-aware state.

This is the best fallback because it preserves target disambiguation.

### Fallback B: Visual-Only ACT

Use only if task-aware state changes break packaging.

Changes:

- Return to 26-D state.
- Generate visually unambiguous training/eval-like scenes as much as possible.
- Still patch SFP port coverage.

Risk:

- Wrong-target failures remain likely.

### Fallback C: Hand-Engineered Runtime Bias

Use only if learned policy reliably approaches but selects the wrong nearby
port.

Idea:

- Keep learned policy for motion style.
- Add a small task-conditioned correction or initial target bias before entering
  learned insertion.

Risk:

- Can become brittle quickly.
- Must not use ground truth at eval.

### Fallback D: CheatCode-Like Policy Without Ground Truth

Use only if learning fails badly.

Idea:

- Detect the requested target from images and use controller state to execute a
  scripted insertion.

Risk:

- Too much perception work for one week unless target detection is already easy.

## Files Likely To Change

Core implementation:

- `aic_engine/config/gen_random_trials.py`
- `aic_utils/lerobot_robot_aic/scripts/record_dataset.py`
- `aic_utils/lerobot_robot_aic/scripts/audit_dataset.py`
- `aic_example_policies/aic_example_policies/ros/RunACT.py`
- possibly new `aic_example_policies/aic_example_policies/ros/RunACTTaskAware.py`
- `aic_model/aic_model/aic_model.py`
- new `aic_utils/lerobot_robot_aic/lerobot_robot_aic/task_encoding.py`

Tests:

- `aic_utils/lerobot_robot_aic/test/test_recording_utils.py`
- new task-encoding tests, likely under `aic_utils/lerobot_robot_aic/test/`

Docs:

- `docs/cheatcode_dataset_collection.md`
- `docs/cheatcode_training_notes.md`
- this file

## Implementation Checklist

Generator:

- [ ] Add `sfp_port_0` and `sfp_port_1` sampling.
- [ ] Add stratified categorical balancing.
- [ ] Add optional boundary-case generation.
- [ ] Emit summary counts after writing YAML.
- [ ] Test generated YAML loads.

Task encoding:

- [ ] Add shared task encoder helper.
- [ ] Unit-test all known task strings.
- [ ] Unit-test unknown string handling.
- [ ] Document `TASK_DIM=16`.
- [ ] Update `STATE_DIM=42`.

Task metadata:

- [ ] Publish active task from `aic_model`.
- [ ] Subscribe in recorder.
- [ ] Require active task before saving frames.
- [ ] Log task at episode start.
- [ ] Add task coverage to audit output.

Recorder:

- [ ] Append task vector to state.
- [ ] Keep differentiated pose-target action path.
- [ ] Reset pose-delta state at episode end.
- [ ] Keep stale-action skipping.
- [ ] Validate image dimensions.
- [ ] Validate state dimension.

Runtime policy:

- [ ] Append task vector in `prepare_observations`.
- [ ] Load task-aware checkpoint.
- [ ] Avoid hard-coded remote downloads for final submission.
- [ ] Use actual loop dt or documented policy dt.
- [ ] Log task vector once per task.
- [ ] Add action clipping if needed.

Audit:

- [ ] Report state shape.
- [ ] Report task-vector min/max/mean.
- [ ] Report categorical target counts.
- [ ] Flag missing target categories.
- [ ] Flag constant task dimensions in serious datasets.

Training:

- [ ] Train v1 ACT.
- [ ] Evaluate checkpoints offline.
- [ ] Run real eval sweep.
- [ ] Train v2 ACT if v1 works.
- [ ] Save final checkpoint locally.

Packaging:

- [ ] Package primary model.
- [ ] Package fallback model.
- [ ] Verify no network dependency.
- [ ] Verify size.
- [ ] Verify cold start.
- [ ] Verify runtime latency.

## Final Recommendation

The best one-week strategy is:

1. Implement task-aware state conditioning immediately.
2. Fix SFP port diversity immediately.
3. Validate with a 20-episode smoke dataset.
4. Train a compact ACT on 300 clean episodes.
5. Use real eval scoring to close the action-control loop.
6. Scale data only after the first checkpoint runs successfully.
7. Freeze early enough to package and submit without last-minute architecture
   risk.

This gives the highest chance of a high hackathon score because it directly
attacks the two biggest practical failure modes: wrong-target ambiguity and
train/eval action mismatch.

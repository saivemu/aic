# CheatCode LeRobot Dataset Collection

> See [`status.md`](status.md) for the project tracker (which plan is currently
> shipped, scores, and what's left). This page is just the recipe for the
> collection step. The plan that drives it lives in
> [`cheatcode_hackathon_success_plan.md`](cheatcode_hackathon_success_plan.md)
> (original May-10 strategy) or
> [`phase_cdef_replan.md`](phase_cdef_replan.md) (post-Plan-D replan).

This workflow records imitation-learning data by running the provided
`aic_example_policies.ros.CheatCode` policy with `ground_truth:=true` during
training. Challenge rules allow ground-truth simulator state during training,
but ground truth is not available during official evaluation, so this data is
used only to train a policy that consumes normal observations at runtime.

## Current status

This guide is maintained on branch `training/cheatcode-pipeline-clean` in the
worktree:

```bash
/home/saivemu/code/aic-worktrees/cheatcode-training-clean
```

The branch was created from latest `origin/main` and currently contains:

- `record_dataset.py`: passive ROS 2 recorder for `/observations` and
  `/aic_controller/pose_commands`.
- `recording_utils.py`: tested pose-target differencing utilities for converting
  CheatCode `MODE_POSITION` commands into RunACT-style velocity actions.
- `gen_random_trials.py`: randomized AIC engine config generator.
- `audit_dataset.py`: offline LeRobot dataset quality summary.
- `eval_checkpoints.py`: checkpoint imitation-MAE comparison on held-out
  episodes.
- `cleanup_engine_bags.sh`: disk cleanup watcher for long eval runs.

Before a large collection run, patch `gen_random_trials.py` to sample both
`sfp_port_0` and `sfp_port_1`. The qualification docs allow either SFP port,
and the current generator only targets `sfp_port_0`.

For durable context across Codex sessions, see
[`cheatcode_training_notes.md`](./cheatcode_training_notes.md).

## Why this exists

Manual keyboard recording is slow and inconsistent for the cable insertion task.
CheatCode already solves the scripted training task using the plug and port TFs,
so it can generate many consistent demonstrations. The recorder in
[`record_dataset.py`](../aic_utils/lerobot_robot_aic/scripts/record_dataset.py)
runs beside CheatCode and passively records:

- `/observations` for the RunACT-compatible state vector and camera images.
- `/aic_controller/pose_commands` for the commanded Cartesian action.

CheatCode publishes `MotionUpdate` messages in `MODE_POSITION`, where
`msg.velocity` is intentionally zero. The recorder therefore differentiates
consecutive pose targets to synthesize the 7-D action vector expected by RunACT:
linear velocity xyz, angular velocity xyz, and an unused zero pad.

## Recommended next steps

1. Add SFP port diversity in `gen_random_trials.py` by sampling `sfp_port_0` and
   `sfp_port_1`.
2. Generate a smoke config with `20` trials, not a full dataset.
3. Collect into a separate smoke dataset such as `${HF_USER}/aic_act_smoke`.
4. Run `audit_dataset.py`.
5. Visualize several episodes with `lerobot-dataset-viz`.
6. Only after the smoke dataset looks correct, collect the first serious
   `300`-episode dataset.

Suggested first target mixture:

- `60-67%` SFP, because qualification has two SFP-style trials.
- `33-40%` SC, because SC is the generalization trial and should not be
  under-sampled.
- Balanced coverage over all five NIC rails.
- Balanced coverage over both SFP ports after the generator patch.
- Balanced coverage over both SC rails.
- Board pose and rail translation coverage across center and boundary values.
- Grasp jitter around the documented `+/-2 mm` and `+/-0.04 rad` ranges.

## Collection flow

1. Generate a randomized engine config:

   ```bash
   pixi run python aic_engine/config/gen_random_trials.py \
     --n 300 \
     --seed 42 \
     --sfp-fraction 0.67 \
     --out aic_engine/config/random_trials_300.yaml
   ```

2. Start the eval stack with ground truth enabled and the randomized config:

   ```bash
   distrobox enter -r aic_eval -- /entrypoint.sh \
     ground_truth:=true \
     start_aic_engine:=true \
     aic_engine_config_file:=$PWD/aic_engine/config/random_trials_300.yaml
   ```

3. Start the CheatCode policy in a second terminal:

   ```bash
   pixi run ros2 run aic_model aic_model --ros-args \
     -p use_sim_time:=true \
     -p policy:=aic_example_policies.ros.CheatCode
   ```

4. Start the recorder in a third terminal:

   ```bash
   pixi run python aic_utils/lerobot_robot_aic/scripts/record_dataset.py \
     --repo-id ${HF_USER}/aic_act_v1 \
     --root ~/.cache/huggingface/lerobot/${HF_USER}/aic_act_v1 \
     --num-episodes 300 \
     --episode-idle-timeout 2.0 \
     --max-action-age 0.25
   ```

5. Keep disk usage bounded while long eval runs write per-trial bags:

   ```bash
   AIC_CTR=aic_eval AIC_RESULTS_DIR=$HOME/aic_results \
     aic_utils/lerobot_robot_aic/scripts/cleanup_engine_bags.sh
   ```

6. Audit the dataset before training:

   ```bash
   pixi run python aic_utils/lerobot_robot_aic/scripts/audit_dataset.py \
     --dataset-repo-id ${HF_USER}/aic_act_v1 \
     --dataset-root ~/.cache/huggingface/lerobot/${HF_USER}/aic_act_v1 \
     --output-json outputs/aic_act_v1_audit.json
   ```

## Quality gates before training

Check the audit output before spending GPU time:

- `zero_action_fraction` should be low. A high value usually means actions were
  recorded from the zero `velocity` field instead of the differentiated pose
  targets, or the recorder was running while CheatCode was idle.
- `linear_speed_spike_fraction` and `angular_speed_spike_fraction` should be
  near zero. Spikes usually indicate an episode boundary leak or timestamp issue.
- Episode lengths should be consistent with the 20 Hz observation rate and the
  task duration. Very short episodes are discarded by default.
- TCP path length and final displacement should not have obvious outliers.

Keep a held-out slice of episodes for checkpoint selection. For example, record
300 episodes, train on episodes `0..254`, and use `255..299` for
`eval_checkpoints.py`.

## Checkpoint evaluation

After training ACT or Diffusion checkpoints, compare them on held-out episodes:

```bash
pixi run python aic_utils/lerobot_robot_aic/scripts/eval_checkpoints.py \
  --checkpoint-dir outputs/train/act_aic_v1/checkpoints \
  --dataset-repo-id ${HF_USER}/aic_act_v1 \
  --val-episodes 255 256 257 258 259 260 261 262 263 264 \
  --max-frames 600 \
  --output-json outputs/act_aic_v1_checkpoint_eval.json
```

This reports per-action-dimension MAE in physical units, which is useful for
choosing the checkpoint that best imitates the CheatCode trajectories before
running the full scoring pipeline.

## Visualization and upload

Quick local visualization:

```bash
pixi run lerobot-dataset-viz \
  --repo-id ${HF_USER}/aic_act_smoke \
  --root ~/.cache/huggingface/lerobot \
  --mode local \
  --episode-index 0
```

After audit and visual inspection pass, upload the dataset to Hugging Face:

```bash
pixi run hf auth login
pixi run hf repo create ${HF_USER}/aic_act_v1 --type dataset --private
pixi run hf upload ${HF_USER}/aic_act_v1 \
  ~/.cache/huggingface/lerobot/${HF_USER}/aic_act_v1 \
  . \
  --repo-type dataset
```

Keep datasets private while iterating.

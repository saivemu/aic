# CheatCode Training Notes

These notes preserve context for future work on the CheatCode-based LeRobot
dataset pipeline.

## Repository state

- Persistent worktree:
  `/home/saivemu/code/aic-worktrees/cheatcode-training-clean`
- Branch: `training/cheatcode-pipeline-clean`
- Base: latest `origin/main` when the branch was created.
- First pipeline commit:
  `911fc7c training: add CheatCode dataset collection pipeline`
- Original active checkout was left on `submit/act-plan-b-trained`.

## Why this branch exists

The old `training/act-cheatcode-pipeline` branch contained useful collection
tooling, but it was not ready to merge as-is. It had personal defaults, stale
planning notes, and the critical CheatCode bridge logic was not isolated for
testing. This branch ports the reusable pieces onto current `origin/main` and
cleans them up for a mainline PR.

## Important implementation details

- CheatCode publishes Cartesian `MotionUpdate` messages in
  `TrajectoryGenerationMode.MODE_POSITION`.
- In that mode, `msg.velocity` is zero. Recording `msg.velocity` directly would
  train a do-nothing model.
- The recorder computes actions by differentiating consecutive pose targets:
  linear velocity xyz, angular velocity xyz, and one unused zero pad.
- Pose-delta state resets at episode boundaries to avoid large inter-trial
  velocity spikes.
- The recorder skips frames whose latest action is stale, controlled by
  `--max-action-age`.

## Current tooling

- `aic_engine/config/gen_random_trials.py`
  Generates randomized trial YAML files for AIC engine collection.
- `aic_utils/lerobot_robot_aic/scripts/record_dataset.py`
  Records RunACT-compatible LeRobot datasets from observations and commands.
- `aic_utils/lerobot_robot_aic/scripts/audit_dataset.py`
  Summarizes action statistics, zero-action fraction, speed spikes, episode
  lengths, and TCP path statistics.
- `aic_utils/lerobot_robot_aic/scripts/eval_checkpoints.py`
  Evaluates ACT or Diffusion checkpoints against held-out LeRobot episodes.
- `aic_utils/lerobot_robot_aic/scripts/cleanup_engine_bags.sh`
  Keeps long eval runs from filling disk with per-trial bag directories.

## Known gap before large collection

Patch `gen_random_trials.py` so SFP trials sample both `sfp_port_0` and
`sfp_port_1`. Qualification docs allow either target, but the current generator
only targets `sfp_port_0`.

## First collection target

Start with a smoke dataset, not a full training run:

```bash
cd /home/saivemu/code/aic-worktrees/cheatcode-training-clean
export HF_USER=saivemu

pixi run python aic_engine/config/gen_random_trials.py \
  --n 20 \
  --seed 1 \
  --sfp-fraction 0.65 \
  --out aic_engine/config/random_trials_smoke.yaml
```

Collect into `${HF_USER}/aic_act_smoke`, audit it, and visualize a few episodes.
Only scale to 300+ episodes after action statistics and video playback look
correct.

## Data mixture for high score

- Use roughly `60-67%` SFP and `33-40%` SC.
- Balance SFP across all five NIC rails.
- Balance SFP across both `sfp_port_0` and `sfp_port_1` after the generator
  patch.
- Balance SC across both SC rails.
- Cover board pose center and boundaries for x, y, and yaw.
- Cover rail translation and yaw boundaries.
- Include grasp jitter around `+/-2 mm` and `+/-0.04 rad`.
- Keep held-out episodes that stress boundaries and target combinations, not
  just a random split.

## Quality gates

Do not train on a dataset until:

- `zero_action_fraction` is low.
- linear and angular spike fractions are near zero.
- recorder logs show low stale-action frames and zero unsupported commands.
- episodes visually show the correct target, smooth approach, and insertion.
- known failed or wrong-target episodes are removed or excluded.

## Model risk

Current RunACT inference logs the task but does not feed task fields into the
model input. If multiple plausible ports are visible, a pure visual ACT policy
may not know which target the engine requested. For higher scores, either make
the policy task-aware or ensure collection/evaluation setups make the target
visually unambiguous.

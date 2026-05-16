# Flow Matching Attempt - 2026-05-15

Goal: test a compact rectified-flow final-stage policy on top of the known best
ACT checkpoint, then push an ECR image only if it beats the current `148.903`
local score.

## Implementation

- Added `lerobot_robot_aic.flow_policy`: a small image/state-conditioned
  rectified-flow action chunker.
- Added `scripts/train_flow_policy.py` for LeRobot AIC datasets.
- Added optional `RunACT` handoff controlled by `AIC_FLOW_POLICY_*` env vars.
- Added `Dockerfile.flow_checkpoint_submission` for packaging ACT base plus a
  flow final-stage helper if a future run beats the ACT-only image.

## Training

Dataset:

- `saivemu/aic_final_recovery30_scscaled_clean25_mixedfull1_final180x4_v1`
- root:
  `/home/saivemu/aic_runs/final_recovery30_scscaled_v1/weighted_mixedfull1_final180x4_lerobot`
- 125 episodes, 39060 frames

Run:

- output:
  `/home/saivemu/code/aic/outputs/train/flow_mixedfull1_final180x4_v1`
- steps: 1500
- checkpoints: 500, 1000, 1500
- chunk length: 16
- flow steps: 4
- replan every: 4
- runtime mode tested: `xy_down`

## Local Scores

Base ACT checkpoint for every run:
`outputs/train/act_final_recovery30_scscaled_clean25_v1/checkpoints/010000/pretrained_model`

| Flow checkpoint | Start | Total | Notes |
| --- | ---: | ---: | --- |
| 000500 | 8s | 110.647 | No insertions; early handoff hurt approach. |
| 001000 | 8s | 116.959 | No insertions; same failure mode. |
| 001500 | 12s | 141.458 | Trial 1 partial insertion, trial 3 regressed. |
| 001500 | 14s | 140.879 | Trial 2 partial insertion, trial 3 regressed. |
| 001500 | 16s | 143.195 | Best flow variant, still below ACT baseline. |
| 001500 | 20s | 137.072 | Later handoff reduced partial insertion quality. |

## Decision

No flow variant beat the current best ACT-only score of `148.903`, so no new ECR
image was pushed. The current best pushed image remains:

`973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:act-clean25-1489-v1`

Digest:
`sha256:2e98cf422fb57f70239f33c7b48e67d7efebfebfa352689b90b24d9cc2ce26d8`

Bottom line: the flow policy can sometimes improve one final insertion case,
but the current dataset/model/runtime handoff is not reliable across the three
scoring trials. The handoff timing dominates the result more than training from
500 to 1500 steps.

# SC Routed Expert Attempt - 2026-05-16

## Goal

Test whether trial 3 improves if the SC geometry is handled by a specialist
ACT policy instead of the shared SFP/SC policy.

## Code Changes

- Added optional `AIC_SC_POLICY_PATH` support in `RunACT.py`.
- The router uses the SC specialist when `task.plug_type` is in
  `AIC_SC_POLICY_PLUG_TYPES` (default: `sc`).
- Added separate SC checkpoint normalization and action unnormalization.
- Gated late insert/flow helpers with the existing
  `AIC_FINAL_HELPER_PLUG_TYPES` setting.
- Added `Dockerfile.routed_sc_submission` for packaging a base policy plus an
  SC policy.
- Added `SC_POLICY_PATH` support to `score_policy.sh`.

## SC Specialist Training

Run:

`act_sc_exact_success8_from_clean25_v1`

Source dataset:

`/home/saivemu/aic_runs/exact_noperturb30_v1/lerobot`

Episodes:

`[2, 8, 11, 17, 20, 23, 26, 29]`

These are the successful SC episodes from the exact no-perturb 30-trial run.

Base checkpoint:

`outputs/train/act_final_recovery30_scscaled_clean25_v1/checkpoints/010000/pretrained_model`

Training command used direct LeRobot episode filtering instead of repacking:

```bash
pixi run lerobot-train \
  --dataset.repo_id=saivemu/aic_exact_noperturb30_v1 \
  --dataset.root=/home/saivemu/aic_runs/exact_noperturb30_v1/lerobot \
  --dataset.episodes='[2,8,11,17,20,23,26,29]' \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.pretrained_path=/home/saivemu/code/aic/outputs/train/act_final_recovery30_scscaled_clean25_v1/checkpoints/010000/pretrained_model \
  --policy.device=cuda \
  --output_dir=/home/saivemu/code/aic/outputs/train/act_sc_exact_success8_from_clean25_v1 \
  --job_name=act_sc_exact_success8_from_clean25_v1 \
  --batch_size=1 \
  --steps=3000 \
  --save_freq=500 \
  --eval_freq=0 \
  --num_workers=0 \
  --wandb.enable=false \
  --policy.push_to_hub=false
```

## Scoring Results

Baseline to beat:

`score_act_final_recovery_clean25_010000_sample3_v1`: `148.903`

| Run | Total | Trial 1 | Trial 2 | Trial 3 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `score_routed_sc_success8_000500_v1` | `126.004` | `43.550` | `43.588` | `38.867` | Whole-trial SC router, 500-step specialist. No insertion. |
| `score_routed_sc_success8_003000_v1` | `126.811` | `43.526` | `43.530` | `39.754` | Whole-trial SC router, 3000-step specialist. No insertion. |
| `score_sc_insert_success8_003000_start8_v1` | `121.816` | `43.551` | `43.702` | `34.563` | Base ACT approach, SC specialist as final insert helper. Trial 3 regressed to 0.08 m final distance. |

## Diagnosis

The router itself works: logs show trial 3 routes to the SC specialist.

The policy does not solve trial 3. The SC specialist reproduces the aggressive
teacher approach, saturates the 20 mm target clamp early, and still never
produces the final insertion event. In final-helper mode, the SC specialist's xy
commands are not reliable under the base policy rollout and push the plug farther
from the port.

The useful result is negative: a model trained only on successful exact SC
episodes is not enough. Trial 3 needs SC recovery data from the actual drifted
states reached by our policy, with verified insertion from those states, or a
more direct port-localization/controller method.


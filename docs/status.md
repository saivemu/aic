# Project status — AIC cable insertion

Single source of truth for what's been done, what's shipped, and what's still open. Last updated 2026-05-14. If anything below disagrees with reality, fix this doc.

**Companion docs you'll want open:**
- [`cheatcode_hackathon_success_plan.md`](cheatcode_hackathon_success_plan.md) — original strategy doc (Phases A–F). Now superseded for C–F by `phase_cdef_replan.md`, but Phases A/B are still valid background.
- [`cheatcode_dataset_collection.md`](cheatcode_dataset_collection.md) — the operational recipe for running the eval-stack + CheatCode + recorder pipeline. The how-to behind the dataset rows in the tracker below.
- [`cheatcode_training_notes.md`](cheatcode_training_notes.md) — implementation details for the recorder + training side (recorder QoS, action synthesis, episode boundaries).
- [`visual_servo_experiment_log.md`](visual_servo_experiment_log.md) — May-14 final-alignment experiments, scores, dead ends, and next steps.
- See the full doc map at the bottom.

## Tracker at a glance

| Plan | Architecture | Dataset | State | Image | Compose total | Status |
|---|---|---|---|---|---|---|
| **A** | ACT, 50k steps | `aic_act_v1` (100 ep) | 26-D | 288×256 | **43.89** | superseded |
| **B** | ACT, 40k steps (val-split) | `aic_act_v1` (300 ep) | 26-D | 288×256 | **112.90** ¹ | superseded by D |
| **C** | Diffusion, 40k steps | `aic_act_v1` (300 ep, same as B) | 26-D | 288×256 | **86.03** | preserved, not deployed |
| **D** | ACT, 40k steps | `aic_act_v2` (299 ep, ep 190 excluded) | 43-D | 576×512 | **123.06 / 123.38 / 123.58** ² | **shipped to ECR 2026-05-12** |
| **E** | Plan D + optional final visual servo/search/insertion handoff | local TF-labeled visual-servo data | 43-D | 576×512 | **124.85 best local** ³ | experimental, not shipped |

¹ Plan B as recorded May-5 from image `aic-runact:plan-b-v3`. Rebuilding the same source today gets 103.6 due to a cuDNN kernel-selection drift; this is the rebuild-variance floor we used to set Plan D's 115 ship gate.
² Plan D: min over 3 back-to-back compose runs is 123.06; variance 0.5 pts.
³ Plan E best is the xy-direction visual-servo branch. It improved local proximity score slightly but did not create contact or insertion, so it is not yet a >150 path.

## What's currently shipped

- **ECR image:** `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:plan-d-v1`
- **Digest:** `sha256:0be3ba70a5acea742f3660a8a9822fbcddba3d0bae1e09f6cb46ce21339c0e72`
- **Manual step pending (you):** paste that URI into the AIC submission portal.

The image also bakes Plan B at `/opt/policy_b` as a fallback; flip with `AIC_POLICY_PLAN=b` in `docker/docker-compose.yaml`.

## May-14 final-alignment experiments

The branch now has optional scaffolding for three final-stage ideas: a guarded
search, an insertion-only policy handoff, and a learned visual servo. The default
runtime remains Plan D unless the new environment variables are set.

Best result from this round is **124.85** with a discrete `xy_direction` visual
servo model starting late and keeping Plan D's z behavior. It did **not** get
contact or insertion. A 45 s trial cap dropped score to **112.56**, so the
speed lesson is explicit: do not spend extra time unless the final controller
reliably converts into contact/insertion.

Full details, artifact paths, commands, and the recommended next order are in
[`visual_servo_experiment_log.md`](visual_servo_experiment_log.md).

## Datasets on disk / Hub

| Repo id | Episodes | State dim | Image res | Use |
|---|---|---|---|---|
| `saivemu/aic_act_v1` | 300 | 26 | 288×256 | training for Plans A/B/C |
| `saivemu/aic_act_smoke_v2` | 20 | 43 | 576×512 | smoke for Plan D schema |
| `saivemu/aic_act_v2` | 300 | 43 | 576×512 | training for Plan D (ep 190 excluded → 299 used) |

## Trained models on disk / Hub

| Local path | HF mirror | Notes |
|---|---|---|
| `outputs/plan_b/pretrained_model/` | `StrivingBapan/aic_act_v1_planb_300ep` | Plan B step-40k |
| `outputs/plan_c/pretrained_model/` | — | Plan C step-40k |
| `outputs/plan_d/pretrained_model/` | — | Plan D step-40k; also at training/checkpoints/{010000,020000,030000,040000,last} in the pixi env |

## What we know about the failure modes

From [`analysis/alignment_learnability.py`](../analysis/alignment_learnability.py) (Plan B-era):

- Plan B's xy alignment MAE is **at the k-NN noise floor on the 26-D state**. No amount of additional 26-D data could have moved it.
- Adding wrist wrench (6-D) gives contact-stage discrimination the original state lacked. Plan D's T1 jump from ~25 → 43 is almost certainly attributable to this.

From the compose run breakdowns (across Plan B and Plan D):

- **No trial inserts.** Final plug-port distance is 0.05–0.09 m on T1/T2, 0.07 m on T3. tier_3 caps at ~30 without insertion; the full insertion bonus is ~75.
- **Plan D's gain is robustness, not capability.** T1 went from sometimes-contacting to never-contacting. T2 was already at ceiling. T3 was a coin-flip; Plan D makes it reliably clean.
- The last-cm alignment problem persists. The model gets close, then commands near-zero velocity, then time runs out.

## What's left to try (ranked by expected leverage)

These are taken from [`docs/phase_cdef_replan.md`](phase_cdef_replan.md) and trimmed to what's still relevant after Plan D shipped. Score deltas are *vs Plan D's 123.06* now, not vs Plan B.

1. **Hybrid: ACT for gross motion, fingertip-CV alignment for the last 3 cm** — Δ +30 to +75 (insertion bonus is +50 per trial). 6–10 hr engineering. High risk (perception code), but only realistic path to actually inserting.
2. **Image scale 0.5 → 1.0** (576×512 → 1152×1024) with resnet18 backbone — Δ +5 to +20. Doubles port-feature pixel density again. ~4 hr collect + 4 hr train. Need to measure inference latency first (open question; was deferred).
3. **resnet18 → resnet34** — Δ +0 to +10. 2 hr. Latency/size risk.
4. **Diffusion v2 on `aic_act_v2`** — unclear delta. Plan C on v1 lost to Plan B by 23%, but Plan C's val MAE was methodologically biased; rerunning with the augmented state and the consistent comparison protocol may surprise.
5. **L2 loss retrain on existing data** — Δ +5 to +15 historically. Cheapest test (2 hr). Likely *less* useful now that Plan D already moves the alignment-noise-floor needle via better state.

The submission slot is daily; Plan D used today's slot. Don't burn tomorrow's unless one of the above clears the +3-pt rebuild-variance gate over Plan D.

## What's NOT worth trying again

- More episodes of the same kind (same 26-D or 43-D state, same images). The val-MAE wash from Plan A → Plan B at 3× data on the same schema is the lesson here.
- Per-tick chunking changes to RunACT. Verified in this round that LeRobot ACT's `select_action` semantics is correct under our `temporal_ensemble_coeff` (Plan B) and `n_action_steps=100` chunk-replay (Plan D) configs.
- Re-architecting the RunACT control loop. The sim-time loop + clamped position-mode + force-aware backoff combo from `edd7f41` works fine; the bottleneck is upstream (model output), not downstream (controller pipeline).

## Pixi-env quirks worth knowing about

These will bite if anyone rebuilds the environment from scratch:

1. **`torchcodec` is disabled** (renamed `*.disabled` in site-packages). The pixi-locked `ffmpeg 8.0.1` ships `libavutil.so.60`, but `torchcodec 0.5` requires `libavutil.so.56–59`. Disabling it lets lerobot fall back to its pyav backend. A future `pixi reinstall` will undo this; either redo the rename or fix the lockfile.
2. **`lerobot_robot_aic.__init__.py` is lazy.** Eager submodule imports tripped a libtiff/libjpeg ABI mismatch under lerobot-train. The package now defers cv2-dependent imports via `__getattr__`.
3. **Recorder `--trials-config` is mandatory.** It maps per-episode attempt-index to task identity (for the Plan D one-hot). Old v1-era recording does not require it.

## Doc map

| Doc | What's in it |
|---|---|
| [`status.md`](status.md) | This page — the single tracker |
| [`overnight_progress.md`](overnight_progress.md) | Hour-by-hour log of the compose-regression investigation + Plan D ship, 2026-05-11 / 12 |
| [`three_way_comparison.md`](three_way_comparison.md) | Plan A vs B vs C deep dive (Phases 1–8). Pre-Plan-D. Still the definitive Plan A/B/C reference. |
| [`cheatcode_hackathon_success_plan.md`](cheatcode_hackathon_success_plan.md) | Original Phase A–F plan (May 10). Mostly historical now; replaced by `phase_cdef_replan.md` for the Phase C–F portion. |
| [`phase_cdef_replan.md`](phase_cdef_replan.md) | Updated Phase C–F plan (May 11) after the regression investigation; what Plan D actually did. |
| [`cheatcode_dataset_collection.md`](cheatcode_dataset_collection.md) | How to drive the eval-stack + CheatCode + recorder pipeline. |
| [`cheatcode_training_notes.md`](cheatcode_training_notes.md) | Implementation notes for the recorder/training side. |
| [`visual_servo_experiment_log.md`](visual_servo_experiment_log.md) | May-14 visual-servo/final-alignment experiment log and next-step order. |

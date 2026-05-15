# Project status — AIC cable insertion

Single source of truth for what's been done, what's shipped, and what's still open. Last updated 2026-05-14 (overnight round). If anything below disagrees with reality, fix this doc.

**Companion docs you'll want open:**
- 🔴 **[`resume_state_2026_05_14_evening.md`](resume_state_2026_05_14_evening.md)** — pinned resume context after the 2026-05-14 evening workstation move. Read this first if returning from a power-off.
- [`overnight_2026_05_14_progress.md`](overnight_2026_05_14_progress.md) — the 2026-05-14 overnight log: 14 configs, ASSIST-mode breakthrough, both ECR images pushed.
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
| **D** | ACT, 40k steps | `aic_act_v2` (299 ep, ep 190 excluded) | 43-D | 576×512 | **123.06 / 123.38 / 123.58** ² | **shipped to ECR 2026-05-12, still the live submission** |
| **E** | Plan D + optional final visual servo/search/insertion handoff | local TF-labeled visual-servo data | 43-D | 576×512 | **124.85 best local** ³ | experimental, not shipped |
| **F-safe** | Plan D + pixel_delta VS in ASSIST mode + z-stiffness 500 N/m boost during VS | same as E | 43-D | 576×512 | **127.77 mean / 128.41 max (5-run)** ⁴ | **pushed to ECR 2026-05-14 as `assist-pixel-zstiff-v1`**, not yet submitted |
| **F-aggressive** | Plan D + pixel_delta VS in REPLACE mode | same as E | 43-D | 576×512 | **124.5 mean / 139.02 max (3-run)** ⁵ | **pushed to ECR 2026-05-14 as `plane-pixel-v1`**, not yet submitted |

¹ Plan B as recorded May-5 from image `aic-runact:plan-b-v3`. Rebuilding the same source today gets 103.6 due to a cuDNN kernel-selection drift; this is the rebuild-variance floor we used to set Plan D's 115 ship gate.
² Plan D: min over 3 back-to-back compose runs is 123.06; variance 0.5 pts.
³ Plan E best is the xy-direction visual-servo branch. It improved local proximity score slightly but did not create contact or insertion, so it is not yet a >150 path.
⁴ Plan F-safe 5 compose runs: 127.25 / 127.49 / 127.67 / 128.03 / 128.41. Tight 1.16 pt spread. **+4.7 over Plan D shipped.** Tier 3 still proximity-only (no full insertions), but ASSIST mode + boosted z stiffness reliably pulls the gripper closer.
⁵ Plan F-aggressive 3 runs: 110.54 / 124.03 / 139.02. **Trial 2 of the 139 run actually scored Tier-3 = 38 (partial insertion)** — the first config tonight to break into the partial-insertion band. 28-pt spread makes it a high-variance bet for a single cluster submission.

## What's currently shipped

The currently submitted entry is still Plan D from 2026-05-12. Two new images
are in ECR and ready to paste into the portal whenever you choose to spend a
daily submission slot:

| Tag | Local eval | Risk | URI suffix |
|---|---|---|---|
| `plan-d-v1` (live submission) | 123.06 cluster | shipped | `…:plan-d-v1` (`sha256:0be3ba70a5acea742f3660a8a9822fbcddba3d0bae1e09f6cb46ce21339c0e72`) |
| `assist-pixel-zstiff-v1` | 127.77 mean / 128.41 max (5-run, 1.16 pt spread) | **low — recommended** | `…:assist-pixel-zstiff-v1` (`sha256:db5fd50759d737718608f3d65ed6757741d335eaaeefe26af7b050daf42c41f4`) |
| `plane-pixel-v1` | 124.5 mean / 139.02 max (3-run, 28 pt spread) | high — one trial hit partial insertion | `…:plane-pixel-v1` (`sha256:6cfdce784add64d55c886e8156e9ee7e496165086d0d09ed73d414385a764141`) |

Full registry prefix: `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:` + the tag above.

Pushed compressed sizes are 5.58 GB each (≈ Plan D + 597 KB for the
visual_servo head and entrypoint shim). The `aic-runact:plans-bc` image
still bakes Plan B at `/opt/policy_b` and Plan D at `/opt/policy_d` so
runtime can switch via `AIC_POLICY_PLAN=b/d`. The Plan F images add
`/opt/visual_servo/best_visual_servo.pt` (the pixel_delta head).

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

- **No reliable trial inserts.** Final plug-port distance is 0.05–0.09 m on T1/T2, 0.07 m on T3 for Plan D. tier_3 caps at ~30 without insertion; the full insertion bonus is ~75.
- **Plan D's gain is robustness, not capability.** T1 went from sometimes-contacting to never-contacting. T2 was already at ceiling. T3 was a coin-flip; Plan D makes it reliably clean.
- The last-cm alignment problem persists. The model gets close, then commands near-zero velocity, then time runs out.

From the 2026-05-14 overnight round (Plan F development):

- **The controller's default 90 N/m × 20 mm position-clamp = 1.8 N max spring force is too soft to actually descend the plug into a port.** Boosting z stiffness to 500–1000 N/m (10–20 N spring force) makes the gripper actually move down. This is what flips Plan E's stuck-at-5cm endgame into observed Z motion. Implemented as the `AIC_VISUAL_SERVO_Z_STIFFNESS` and `AIC_FORCE_DESCENT_Z_STIFFNESS` env vars, both gated to default-off so Plan D stays untouched.
- **Visual-servo `pixel_delta` head dominates `xy_direction` once you ADD instead of REPLACE.** REPLACE-mode pixel_delta is too noisy (3-run range 110–139). ADD-mode (`AIC_VISUAL_SERVO_ASSIST_MODE=1`) uses Plan D's xy as the fallback when the head wobbles, giving tight 1.16 pt variance at +4.7 over Plan D. The classifier-direction head loses to pixel_delta in both modes.
- **The port-localization gap is the real blocker past ~130.** Plan D leaves the gripper 50–90 mm laterally from the port; the trained visual-servo heads only have signal at ≤14 mm offsets (median 4.3 mm in the training set). Forcing the gripper to descend with high z stiffness at the wrong xy lands the plug into the *board surface*, scoring proximity (~22) rather than partial insertion (38+). Concrete fix in the "What's left to try" list above.

## What's left to try (ranked by expected leverage, post-2026-05-14)

Score deltas are *vs Plan D's 123.06* shipped baseline. The May-14 overnight
round established that the **port-localization gap** (we leave the plug 5 cm
laterally from the port) is what's actually capping us — controller mechanics
work (stiffness override → real descent), but we don't know where to descend
*to*. So the leverage ranking changed:

1. **Port-keypoint detector + 3-camera triangulation.** Δ +20 to +75. The 20-ep
   visual-servo dataset already saves TF-projected port pixel locations per
   camera (`record_visual_servo_dataset.py`); train a small per-camera
   heatmap detector, triangulate to base-frame, use that as the NAVIGATE
   target in Plan F's force-descent state machine (already implemented and
   env-var-gated in `RunACT.py`). 6–10 hr. **The only known path to repeated
   actual insertions.** Without it we are stuck at ~130 ceiling.
2. **Re-record visual-servo dataset with long-range deltas.** Δ +5 to +20.
   Current dataset's xy norm is median 4.3 mm, p90 13.7 mm — the heads we
   trained only have signal close to the port. Use the recorder with
   `AIC_CHEATCODE_XY_OFFSET_MAX_M=0.08` for a fresh ~50-ep dataset, then
   retrain `pixel_delta` (and / or the keypoint detector). Should make the
   existing ASSIST-mode pipeline reach partial insertion more reliably.
3. **Image scale 0.5 → 1.0** (576×512 → 1152×1024) with resnet18 backbone — Δ +5
   to +20. Doubles port-feature pixel density again. ~4 hr collect + 4 hr
   train. Inference latency at 1.0 not yet measured; open question.
4. **Diffusion v2 on `aic_act_v2` with Plan F-safe runtime config.** unclear
   delta. Plan C on v1 lost to Plan B by 23%, but Plan C's val MAE was
   methodologically biased; rerunning with augmented state + the new
   pixel_delta-in-ASSIST runtime may produce a multimodal alternative that
   handles the contact-and-retry pattern ACT can't.
5. **L2 loss retrain on existing Plan D data** — Δ +5 to +15 historically.
   Cheapest test (2 hr). Likely *less* useful now that Plan F-safe already
   gets +4.7 from the runtime layer.
6. **SO-101 teleop bridge + diffusion final-stage policy** (was Track 2 in
   the morning plan; blocked overnight by hardware). Δ +10 to +30 on top
   of port-localization. 3–5 days end-to-end.

The daily submission slot is open as of 2026-05-14. The two Plan F images
are in ECR and verified locally — paste either URI into the portal whenever
you want to use a slot. `assist-pixel-zstiff-v1` is the safe pick (+4.7
mean, 1.16 pt spread); `plane-pixel-v1` is the high-ceiling gamble (range
110–139, one trial hit partial insertion in local eval).

## What's NOT worth trying again

- More episodes of the same kind (same 26-D or 43-D state, same images). The val-MAE wash from Plan A → Plan B at 3× data on the same schema is the lesson here.
- Per-tick chunking changes to RunACT. Verified that LeRobot ACT's `select_action` semantics is correct under our `temporal_ensemble_coeff` (Plan B) and `n_action_steps=100` chunk-replay (Plan D) configs.
- Re-architecting the RunACT control loop. The sim-time loop + clamped position-mode + force-aware backoff combo works fine; the bottleneck is upstream (model output) and perception (port localization), not downstream (controller pipeline).
- **Force-feedback spiral / blind perturbation search** (T1.2 v1 in the overnight log, scored 117.5). Walking the gripper sideways without knowing where the port is loses proximity points faster than it gains contact bonus. Use port-localization first.
- **Tightening the ASSIST-mode confidence threshold above ~0.5** for the xy_direction head. The classifier's per-axis confidence rarely clears 0.75; high thresholds gate out the assist 70% of the time, leaving Plan D's stuck-state to dominate. Either use pixel_delta (no per-axis confidence, just continuous output) or drop the gate.
- **Visual-servo direction speed > ~6 mm/s in REPLACE mode** (planE_fast scored 118.5 mean with one 106 run). The head outputs occasional wrong directions; at fast speeds those errors compound into off-port drift.
- **Extending `AIC_MAX_TRIAL_S` past 30 s.** The 2026-05-14 run with 45 s scored 112.56 — Tier-2 duration penalty grows linearly and eats whatever proximity improvement the extra time buys.

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
| [`overnight_2026_05_14_progress.md`](overnight_2026_05_14_progress.md) | May-14 overnight log: 14 configs swept, ASSIST-mode pixel_delta + z-stiffness breakthrough (mean 127.77), both Plan F images pushed to ECR. |

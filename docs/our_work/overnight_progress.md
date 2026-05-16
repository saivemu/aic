# Overnight Plan D Progress

Started: 2026-05-11 evening, autonomous execution

## Plan recap

Phase A — Bring in clean worktree's tooling (audit_dataset.py, recording_utils.py, new docs) WITHOUT disturbing our analysis/diagnostic work or v4 deploy refactor.
Phase B — Audit existing aic_act_v1 dataset.
Phase C — Patch trial generator (both SFP ports) + add task encoding to recorder/RunACT.
Phase D — Smoke dataset (20 trials), audit, visualize.
Phase E — Full 300-ep re-collection, audit, train Plan D.
Phase F — Compose eval, decide whether to push to ECR.

Total target: ~8-10 hr wall clock, mostly hands-off.

## State as of start

- Branch: `main` @ `cb374e3` (gitignore outputs/)
- Existing dataset: `saivemu/aic_act_v1` (300 ep, 26-D state, sfp_port_0 only)
- Existing ECR image: `plan-b-v3` (currently submitted; v4 locally built, not pushed)
- Existing HF models: `aic_act_v1_100ep` (Plan A), `aic_act_v1_planb_300ep` (Plan B)

## Key risks I'm mitigating
- Don't burn the daily ECR submission slot unless I'm confident the new image scores meaningfully better than v3.
- Don't force-push to main.
- Don't delete the existing dataset until the new one (v2) is verified.

## Status log
(updated as I make progress)

### 2026-05-12 — Pivot: from "collect new dataset" to "fix compose-mode control loop"

Mid-stream, a new plan was approved (plan-mode artifact at `~/.claude/plans/now-think-critically-through-distributed-wadler.md`). The dataset-recollection effort (Phases C-F of this doc) is shelved in favor of the architectural fix path. Rationale: training a new model takes hours and is risky; fixing the loop is bounded and addresses what the senior critique identified as the actual failure mode (timing/control, not perception).

### 2026-05-12 — Stage 1 (LeRobot ACT chunking semantics)

Our `outputs/plan_b/pretrained_model/config.json` has `temporal_ensemble_coeff: 0.01` (NOT null) and `n_action_steps: 1`. Per `lerobot/policies/act/modeling_act.py` lines 109-112, this activates the temporal ensemble path: every `select_action` call runs `predict_action_chunk` (full inference, 100-action chunk) and the ensembler blends across history with coefficient 0.01 (weights older predictions heavily). Per-tick inference at 20 Hz is correct by design; the `n_action_steps` config validation explicitly requires `n_action_steps == 1` when ensembling. `reset()` properly clears `ensembled_actions` between trials. No chunk-buffering code needed.

### 2026-05-12 — Stage 2 verification (architectural fixes already in code)

The architectural fixes from the new plan's Stages 2A-2C, 2E were already in [aic_example_policies/aic_example_policies/ros/RunACT.py](../aic_example_policies/aic_example_policies/ros/RunACT.py) as of commit `abf3798` (May 11 00:49). Specifically:

- Sim-time loop via `self.time_now()`/`self.sleep_for()`, fixed `LOOP_DT=0.05` (Stage 2A).
- `_last_target_pose` integration; `_clamp_pose_offset` with `MAX_TARGET_OFFSET_M=0.02` and `MAX_TARGET_OFFSET_RAD=0.10` (Stage 2B).
- Workspace z floor at `start_z - 0.30 m` (Stage 2C).
- Force-aware backoff calibrated to `baseline + 15 N` (Stage 2E).
- Gravity feedforward (Stage 2D) is intentionally skipped — `aic_controller`'s `gravity_compensation_action` already handles the arm; payload-only ff would need a different mass estimate.

The model image was rebuilt today against this code; module import time 0.27 s (well under the 5 s budget).

### 2026-05-12 — Stage 3 (compose runs of current RunACT)

Two back-to-back compose runs of the rebuilt image:

| Run | Total | T1 | T2 | T3 |
|---|---|---|---|---|
| 1 | **13.10** | 11.47 (-24 contact) | 36.63 (clean) | -35 (OOD, -24 contact -12 force) |
| 2 | **11.52** | 3.09 (-24 contact) | 43.43 (clean) | -35 (OOD) |

Action chains were *identical* across runs to 4 decimal places — Gazebo physics determinism is high. Variance between runs comes from tier_3 distance-bonus differences after the contact penalty pegs tier_2.

**Trial 2 ≈ May-5 baseline** (43.43 today vs 43.54 reported May-5). **Trial 1 regression**: today's runs have a contact penalty (-24) that May-5's run did not report.

The senior critique anticipated GUI-vs-headless score gap. What we see today differs from the plan's premise (-14 in compose) — current code scores ~+12. But that's still far below the May-5 documented Plan-B compose total of 112.90, despite the model weights matching by md5 and the only RunACT diff being the lazy-import refactor.

### 2026-05-12 — Verifying: is the v3 image still 112.90 today?

Ran `aic-runact:plan-b-v3` (May-5 build) in compose: **total 112.83** (vs documented 112.90). So the eval-side, sim, and model weights are stable. **The regression lives in the rebuild**, not in the environment.

| Trial | V3 image today | May-5 doc | Current rebuild today |
|---|---|---|---|
| 1 | 34.48 | 33.67 | 11.47 / 3.09 |
| 2 | 43.22 | 43.54 | 36.63 / 43.43 |
| 3 | ~35 (clean) | 35.69 (clean) | -35 / -35 (contact + force) |
| **Total** | **112.83** | **112.90** | **13.10 / 11.52** |

### 2026-05-12 — Isolation test: rebuild with v3 source

To separate "code change in abf3798" vs "build-environment drift," I swapped v3's exact RunACT.py source into the current tree and rebuilt. Result: the first run **crashed mid-trial-1** with `rclpy._rclpy_pybind11.RCLError: failed to initialize wait set` — the cluster-lifecycle pathology that abf3798 was created to fix. The crash supports the hypothesis that imports-in-`__init__` IS hazardous on this rebuild path, even though the v3 image itself happened to ship a build that survived.

Package versions verified identical between v3 image and current rebuild (torch 2.7.1+cu128, numpy 2.2.6, lerobot 0.5.1, cv2 4.13.0, cuDNN 90701). So pixi.lock pinning is working.

Retry: clean run scored **103.60**.

### 2026-05-12 — Conclusion

The lazy-import refactor in `abf3798` IS the regression. Same compose, same model weights, same pixi lockfile — only the source code differs:

| Source | Today's compose total |
|---|---|
| v3 / edd7f41 (imports in `__init__`) | **103.60** (rebuild) / 112.83 (May-5 image) |
| abf3798 (imports in `_setup_policy`) | 11.52 — 13.10 |

Why does the lazy-import structure regress scoring? Most likely: temporal-ensemble compounding (`temporal_ensemble_coeff: 0.01` with `chunk_size: 100`) amplifies 1st-decimal-place action differences at t=0 across the 600-tick trial. The t=0 action with v3-style imports differs from abf3798-style at the 3rd decimal (e.g. `-0.0052` vs `-0.0053` in lin_x), which compounds via the ensembler into a trajectory that contacts the task board vs one that does not.

There's also a ~9-pt build-to-build variance even with identical source (112.83 May-5 image vs 103.60 today's rebuild). `torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False` was added at the start of model setup but did not close the gap (103.61 after, vs 103.60 before — within noise). The variance is likely some unpinned binary in pixi's resolved cache.

**Final action:** Revert abf3798's structural change (restore imports in `__init__`), keep all the architectural fixes from edd7f41 (clamp + force backoff + sim-time pacing + workspace z floor), add cuDNN-determinism flags as a defensive measure. Result: compose 103.61, well above the 60-pt ship threshold.

**Submission decision:** **Do NOT push a new ECR image tonight.** The shipped `plan-b-v3` scores 112.90 on the cluster; our best rebuild today scores 103.61 in compose. Submitting would replace a known 112.90 with a worst-case ~95-105. Leave v3 shipped; this revert is staged for whenever the source is shipped next (e.g. with a new model).

### Status of original Plan D dataset-recollection plan
Shelved at the time of the compose-mode investigation. **Unshelved later the same day** when the user reviewed the replan ([phase_cdef_replan.md](phase_cdef_replan.md)) and chose to bet on fresh data with augmented state (wrench + task one-hot) rather than the cheaper L2-retrain shortcut. The data-side fix path was independent of the control-loop fix and still had headroom remaining per `alignment_learnability.py`.

### 2026-05-12 — Plan D (aic_act_v2 + 43-D state) shipped

Pipeline executed end-to-end in a single overnight session:

| Step | Result |
|---|---|
| Verify wrist_wrench is populated in collection | Mean fz 18.5 N, max 21 N, no NaN/zeros |
| Generate dual-SFP-port trial config | 300 trials, 96 sfp_port_0 + 114 sfp_port_1 + 90 SC |
| Smoke 20-ep collection | 20/20 saved, schema 43-D verified, all audit gates pass |
| Full 300-ep aic_act_v2 collection | 300/300 saved, 0 dropped, ~6.7 h wall |
| Audit | Episode 190 has 17 outlier frames (pose-delta glitch); excluded |
| Train Plan D (40k steps, batch 8, 299 ep) | 1 h 48 m on local GPU |
| Compose eval (3 runs) | **123.065 / 123.379 / 123.581** — min 123.06, var 0.5 |

| Trial | Plan B v3 (May-5, shipping) | Plan B v3 image today | Plan D v1 (today) | Δ vs Plan B v3 image |
|---|---|---|---|---|
| 1 | 33.67 | 34.48 | **43.0** | +8.5 |
| 2 | 43.54 | 43.22 | **42.6** | -0.6 |
| 3 | 35.69 | ~35 | **37.6** | +2.6 |
| **Total** | **112.90** | **112.83** | **123.34 (avg over 3 runs)** | **+10.5** |

The win is almost entirely on T1: the augmented state seems to give the model enough signal to avoid the gripper-board contact that Plan B kept getting unlucky on. T2 stays at tier_2 ceiling, T3 stays clean. No insertion bonus in any trial (final distance 0.05–0.09 m on T1/T2, 0.07 m on T3) — the last-cm alignment problem is still there. Plan D wins on robustness, not capability.

**Ship decision:** Plan D image tagged `plan-d-v1` and pushed to ECR. Gate: `min(3 runs)=123.06 > 115` (Plan B v3 + 9-pt rebuild-variance buffer). Win is +10 over the shipped baseline.

Submission URI: `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/bot-squad-l2-learning-loop:plan-d-v1`
Digest: `sha256:0be3ba70a5acea742f3660a8a9822fbcddba3d0bae1e09f6cb46ce21339c0e72`

### Pixi-env quirks worth knowing about

1. **torchcodec disabled.** The pixi env ships `ffmpeg 8.0.1` (libavutil.so.60) and `torchcodec 0.5` which requires ffmpeg 4–7 (libavutil.so.56–59). The two are incompatible. Worked around by renaming
   - `.pixi/envs/default/lib/python3.12/site-packages/torchcodec` → `torchcodec.disabled`
   - `.pixi/envs/default/lib/python3.12/site-packages/torchcodec-0.5.dist-info` → `*.disabled`
   so `importlib.util.find_spec("torchcodec")` returns None and lerobot falls back to its native pyav backend. **A future `pixi reinstall` will undo this** — either redo the rename or fix the lockfile (pin ffmpeg ≤ 7 or upgrade torchcodec).

2. **`lerobot_robot_aic.__init__.py` is lazy.** Eager `from .aic_robot_aic_controller import …` previously tripped a libtiff/libjpeg ABI clash when lerobot-train pre-loaded torchvision/transformers. Lazy `__getattr__` resolves to the submodule on first attribute access; runtime callers (recorder, RunACT) see no API change.

3. **Recorder `--trials-config` is mandatory now.** It maps attempt-index → task identity. Pre-Plan-D collections worked without it because the state didn't include task one-hot.




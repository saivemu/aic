# Phases C–F replan after the 2026-05-11 control-loop investigation

This doc replaces the original [cheatcode_hackathon_success_plan.md](cheatcode_hackathon_success_plan.md) Phases C–F with a sharper plan informed by tonight's compose-mode regression investigation. It is the result of asking "given what we now know, what would actually break 112.9?"

## Tonight's outcome (re-grounding)

The compose-mode regression we'd been chasing (−14 vs 90 in GUI) turned out to be a build-environment artifact in commit `abf3798` (lazy-import refactor), **not** a model/data problem. Reverted in `b46215e`. Compose rebuild now scores 103.6; shipped `plan-b-v3` still 112.83 in compose / 112.90 on cluster today. ECR slot intact.

Crucially, [analysis/alignment_learnability.py](../analysis/alignment_learnability.py) already established that Plan B is **at the k-NN noise floor on xy** given the current 26-D state vector. State-only modeling has zero remaining headroom on lin_x/lin_y at insertion altitude. More data of the same kind will not help — only new *information* (additional observation channels, higher-bandwidth perception) or a different inductive bias for the last 5 cm can move the needle.

## Phase-by-phase verdict

### Phase C — Patch trial generator + add Task encoding to recorder/RunACT
**Revise, don't drop.** The two-port coverage fix in [aic_engine/config/gen_random_trials.py](../aic_engine/config/gen_random_trials.py) is cheap and removes an obvious distribution hole (currently hardcoded to `sfp_port_0`). Task one-hot encoding alone does not address the noise floor — the model already implicitly knows which port to approach (clean tier_2 on trials 1 and 2). Task encoding's real payoff is disambiguating near-identical visual scenes, which is not the dominant failure mode at 112.9. Keep the generator patch (~1 hr); ship task encoding as part of the state-augmentation bundle in Phase E.

### Phase D — Smoke dataset (20 ep)
**Keep but rescope.** A 20-episode smoke is now needed not to validate task encoding alone but to validate **the new state vector schema** (wrist_wrench + task one-hot, ~48-D) end-to-end through recorder, audit, and train without surprises. Without a smoke pass, a 5+ hour full collection on a broken state schema is a worst-case loss.

### Phase E — Full 300-ep re-collection + train "Plan D"
**Replace with a bundled "Plan D state-and-perception bundle."** A pure "300 fresh episodes, same 26-D, same L1" run is contraindicated by the noise-floor analysis. The Plan D *training* run is still worth doing, but it must change at least one of three load-bearing variables: state information content, image resolution, loss function. The original plan changed none of them.

### Phase F — Compose eval + ECR decision
**Keep verbatim.** The gate logic is right: compose-eval the new model on the same seeds; only push to ECR if it clears >115 reliably over ≥3 runs (≥+3 pts headroom vs the shipped 112.90 to absorb the ~9-pt rebuild variance documented in [overnight_progress.md](overnight_progress.md)). One daily slot, one shot.

## What would actually break 112.9?

Ranked by expected leverage. Score-delta estimates are rough.

| # | Candidate | Δ vs 112.9 | Hours | Risk |
|---|---|---|---|---|
| 1 | Add `wrist_wrench` (6-D) to state | +15 to +25 if it helps trial 1/2 insert; 0 if visual is bottleneck | 1 hr code + 5 hr collect + 1 hr train | low |
| 2 | Image scale 0.25 → 0.5 (160×120 → 320×240) + resnet18 | +10 to +30 (pixel/feature density at port doubles) | 2 hr config + 5 hr collect + 2 hr train | medium (latency, mem) |
| 3 | Hybrid: ACT for gross motion, fingertip CV alignment for last 3 cm | +30 to +75 (insertion bonus is +50 per trial) | 6–10 hr | high (perception code) |
| 4 | L2 loss retrain on existing data | +5 to +15 | 1 hr config + 1 hr train | low |
| 5 | Add sfp_port_1 + task one-hot | +5 if hidden eval uses port_1; 0 otherwise | 3 hr code + 5 hr collect + 2 hr train | low |
| 6 | Diffusion v2 (Plan C re-trained correctly) | unclear; Phase 5 eval was biased | 1 hr train + 0.5 hr eval | medium |
| 7 | resnet18 → resnet34 | +0 to +10 | 2 hr | medium (size/latency) |

The top three are the only ones that plausibly clear the +3-pt rebuild-variance gate.

## Recommended next-session sequence

Decision made: skip the L2-retrain shortcut, go straight to a bundled fresh data collection (#1 + #2 + #5 together).

1. **Verify `wrist_wrench` is actually populated** during CheatCode-driven dataset collection. RunACT reads it ([RunACT.py:378](../aic_example_policies/aic_example_policies/ros/RunACT.py)) but [record_dataset.py](../aic_utils/lerobot_robot_aic/scripts/record_dataset.py) does not currently log it. Spot-check one bag from the existing `saivemu/aic_act_v1` dataset before designing state schema around it. **If wrench is silent during collection, the highest-leverage lever (#1) collapses** — replan from there.
2. Patch [aic_engine/config/gen_random_trials.py](../aic_engine/config/gen_random_trials.py) for both SFP ports (~50/50) and write a shared `task_encoding.py` helper.
3. Modify recorder schema:
   - `STATE_DIM = 26 + 6 (wrist_wrench) + 16 (task one-hot) = 48`
   - `image_scaling = 0.5` (320×240, matched at training and inference)
4. Modify [RunACT.py](../aic_example_policies/aic_example_policies/ros/RunACT.py) to (a) extract task identity from the `Task` struct in `insert_cable()` and prepend the same encoding to state, (b) use the new image scale.
5. **Phase D smoke**: 20 trials, run [audit_dataset.py](../aic_utils/lerobot_robot_aic/scripts/audit_dataset.py), confirm:
   - wrist_wrench mean/std in normal range, no nan/zero columns
   - task encoding vectors match generator config order
   - episode lengths and zero-action fraction within prior gates
6. **Phase E full collect**: 300 ep with new distribution (200 SFP balanced over 2 ports × 5 rails, 100 SC). Use the empirical knowledge of GZ rate decay — restart Gazebo every ~80 episodes.
7. **Train Plan D** with bundled state + image scale. Targets: train wall-clock <2 hr, model size still <100 MB for the inference budget.
8. **Phase F gate**: 3 back-to-back compose runs, all same seeds. Push to ECR iff `min(run) > 115`. Otherwise hold the slot, iterate.

## Open questions (deferred answers; will refine plan when resolved)

1. ~~Time budget~~ → "No issue. Bet on Phase E and get clean data." (user, 2026-05-11)
2. Does the hidden qualification eval actually use `sfp_port_1`? Currently unknown. Including port_1 in collection is cheap insurance, so default YES.
3. Inference latency at image_scale=0.5 + resnet18 — must measure before training Plan D. If per-step inference exceeds 40 ms on the eval GPU, fall back to 0.375 or 0.25.
4. **Is `wrist_wrench` populated correctly during dataset collection?** Pre-collection verification step.

## Critical files for the implementation pass

- [aic_engine/config/gen_random_trials.py](../aic_engine/config/gen_random_trials.py) — generator
- [aic_utils/lerobot_robot_aic/scripts/record_dataset.py](../aic_utils/lerobot_robot_aic/scripts/record_dataset.py) — recorder state schema
- [aic_example_policies/aic_example_policies/ros/RunACT.py](../aic_example_policies/aic_example_policies/ros/RunACT.py) — inference state schema (must match training)
- [aic_interfaces/aic_model_interfaces/msg/Observation.msg](../aic_interfaces/aic_model_interfaces/msg/Observation.msg) — verify wrist_wrench shape
- [aic_interfaces/aic_task_interfaces/msg/Task.msg](../aic_interfaces/aic_task_interfaces/msg/Task.msg) — task struct for encoding
- [analysis/alignment_learnability.py](../analysis/alignment_learnability.py) — noise-floor reference; rerun after Plan D training to verify the new state vector actually moved the floor.

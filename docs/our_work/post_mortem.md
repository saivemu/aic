# Post-Mortem: AIC Cable-Insertion Challenge

_Reconstructed 2026-06-29 from the code, the `docs/our_work/` notes, and `outputs/.../results/*_summary.txt`._

## TODO — the four levers to try (in priority order)

- [ ] **1. Stop cloning the oracle blind — go DAgger-style.** Roll out the student policy, collect CheatCode oracle corrections from the *drifted* 50–90 mm states it actually reaches at eval, and train on those. This is the "recovery dataset" idea from `final_approach_recovery_dataset_plan.md` — it needs to be move #1, not the fallback. Stop filtering to `tier_3==75` and stop skipping perturbation frames; those are exactly the recovery states the policy lacks.
- [ ] **2. Localize the port explicitly.** Build a port-keypoint detector + 3-camera triangulation instead of hoping end-to-end BC implicitly learns port localization from a privileged teacher that never had to. Status notes already call this "the only known path to repeated actual insertions."
- [ ] **3. Build a fast online proxy metric.** A handful of short closed-loop sim rollouts scored only on final plug↔port XY distance, to replace the uncorrelated offline MAE and break the ~2-hour compose-gate bottleneck. Without a fast, *correlated* signal, iteration stays statistically underpowered.
- [ ] **4. Run a vision-conditioned learnability test.** Before ever again concluding "no signal," run the kNN/learnability analysis over *image features*, not state alone. The current `alignment_learnability.py` excludes the cameras, so it only proved proprioception can't recover xy — not that vision can't.

---

## Bottom line

The task was never solved. Across ~13 days and a dozen+ approaches, **no policy ever achieved a reliable full insertion** — the one thing the challenge rewards (Tier-3 = +75/trial). Scores plateaued in the *proximity* band: best robust result ~127, best single-run candidate 148.9/300, against a ~200 success target. WaveArm dummy ≈ 39; first ACT (Plan A) = 43.89. Every point came from smoothness/duration/proximity, never from threading the plug.

**The live submission stayed Plan D (~123) the whole time.** Better images were built and locally verified (`assist-pixel-zstiff-v1` @ 127.77, `act-clean25-1489-v1` @ 148.9), pushed to ECR, and never pasted into the portal — even though submissions were free and unlimited per day. Building/scoring/digest-tracking became a substitute for submitting.

## What was built

Behavior cloning of a privileged scripted oracle (**CheatCode**, reads ground-truth port/plug TFs) into a student that sees only cameras + proprioception. On top of that base:

- **Policies:** ACT (Plans A/B/D — workhorses), Diffusion (Plan C, 86), rectified-flow (143), per-task SC "routed expert" (127).
- **Hand-coded last-cm controllers** (all env-var-gated, Plan D as fallback): learned visual-servo head (4 modes), force-descent state machine, z-stiffness override, ASSIST-mode pixel-delta head, pixel-gated insertion controller, spiral search. `RunACT.py` grew to ~2,556 lines holding six of these.
- **Heavy infra:** ECR packaging, lazy-import cold-start budgeting, torchcodec/libavutil ABI workaround, pixi build-cache fights, cuDNN-nondeterminism rebuild variance.

## Where it went wrong

**1. Wrong learning setup — structurally unwinnable as framed.** Cloning a ground-truth oracle with *no recovery data* is the textbook covariate-shift trap. CheatCode flies straight to the port, so demos contain only tiny near-port corrections (median 4.3 mm, ≤14 mm). At test the student drifts to **50–90 mm** lateral error — a state with **zero recovery demos**. Full insertion needs ~5 mm tolerance, so +75 was unreachable by construction. The selection funnel made it worse: kept only `tier_3==75`, oversampled the final window 5×, skipped perturbation frames — training out the recovery behavior needed.

**2. Headline diagnosis was a partial misfire.** "L1-regression-to-median" is correct as a *symptom*, wrong as the binding *cause*. `alignment_learnability.py` reads state + action but **never touches the images**, so its "noise floor" proves only that *proprioception* can't recover xy — not that *vision* can't, and the task is visual port localization. Multimodal heads (diffusion, flow) that don't collapse to the median *also* failed — that negative result was proof it's an **information/observability problem, not a loss problem**. But the loss-shape framing absorbed effort first; the cheap decisive experiments (L2 retrain, vision-conditioned learnability) slid from "Priority 1" to "#5" and were never cleanly run.

**3. Diagnosed the real blocker, then didn't build the fix.** By 05-14 `status.md` names it: "the port-localization gap is the real blocker past ~130," fixable only by a port-keypoint detector + triangulation plus on-policy recovery data. That fix stayed at the **top of "what's left to try" through the final note, never implemented.** Effort went to downstream band-aids — each correctly noted as patching the last centimeter while "the bottleneck is upstream." Best they ever did: one non-reproducible partial insertion (Tier-3 = 38) in a single high-variance run.

**4. Evaluation loop couldn't support the needed iteration.** Only goal-correlated metric was the compose gate: **~2+ hrs per 3-run avg, ±1 to ±28 pt spread, not reproducible across rebuilds** (identical Plan B source: 112.83 → 103.6 from cuDNN kernel drift, a ~9 pt floor). The fast metric (offline action MAE) was uncorrelated (A→B was a MAE wash while score went 43.89 → 112.90) and biased against diffusion by construction. Result: flying blind; many "+4.7" sweep wins sit *inside the noise band* — several configs ranked on 2 runs, some on 1 aborted run, cherry-picked maxima (139.02) quoted while the mean was 124.5.

**5. Time allocation inverted the priorities.** A whole overnight went to an infra ghost — the `abf3798` "compose regression" was a build artifact, not a model problem. ABI fights, ECR digests, cold-start budgeting fill the logs. The data/perception root cause never got its turn.

## What was done well

The diagnostics were better than the decisions. Killed the earlier wrong "regression to tiny actions" thesis after proving the scoring report's `path length: 0.00 m` was a measurement artifact (TICK-logged proof the TCP moved 200–300 mm). Plan B had a proper episode-level val split. Notes are honest about variance and about not spending a submission slot on a worse-than-known image. The "signal-exists-in-data → model-side fix" decision framework is sound. The problem wasn't rigor — it was aim and follow-through.

## Score reference

| Plan | Method | Score | Insertion? |
|---|---|---:|---|
| A | ACT 50k, 26-D | 43.89 | none (T3 collision −35) |
| B | ACT 40k, 300ep, 26-D | 112.90¹ | none |
| C | Diffusion | 86.03 | none (T3 0.19 m out) |
| **D (shipped)** | ACT, 43-D, `aic_act_v2` | **123.06** | none (final 0.05–0.09 m) |
| E | D + visual-servo handoff | 124.85 | none |
| F-safe (ECR, unsubmitted) | D + pixel ASSIST + z-stiff | 127.77 mean | none |
| F-aggressive (ECR) | D + pixel REPLACE | 124.5 mean / 139.02 max | 1 partial (T3=38), 1 trial |
| clean25 (ECR, unsubmitted) | ACT recovery subset | 148.90² | none |
| SC routed expert | base + SC specialist | 126.81 | none (regressed) |

¹ Identical rebuild yields 103.6 (cuDNN nondeterminism, ~9 pt floor). ² Single run, no variance reported.
WaveArm dummy ≈ 39 · theoretical max ≈ 225 · stated success target 200 (never reached).

## Core failure mode (one sentence)

Behavior cloning of a privileged oracle with no recovery coverage produced a policy that cannot localize the port from pixels at the last centimeter — it drives to ~5 cm proximity, commands near-zero velocity, and times out — and the one intervention the analysis pointed to (on-policy recovery data + explicit port localization) was identified, ranked highest-leverage, and never executed.

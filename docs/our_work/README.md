# Our Work

This folder contains our AIC cable-insertion experiment history, not the
official challenge reference docs.

## Current Read Order

| Doc | Purpose |
|---|---|
| [status.md](status.md) | Main tracker for shipped images, datasets, scores, and known failure modes |
| [final_approach_recovery_dataset_plan.md](final_approach_recovery_dataset_plan.md) | Current plan: collect only guaranteed-insertion final-recovery data, then train/evaluate |
| [perturbed_cheatcode_dataset_plan.md](perturbed_cheatcode_dataset_plan.md) | Earlier perturbation-data plan plus the 2026-05-15 smoke validation |
| [visual_servo_experiment_log.md](visual_servo_experiment_log.md) | Final-alignment visual-servo attempts and why they did not reliably insert |
| [overnight_2026_05_14_progress.md](overnight_2026_05_14_progress.md) | Plan F pixel-delta/z-stiffness sweeps and ECR image notes |
| [overnight_progress.md](overnight_progress.md) | Plan D collection/training/eval log |
| [three_way_comparison.md](three_way_comparison.md) | Plan A/B/C comparison and alignment-collapse diagnosis |
| [phase_cdef_replan.md](phase_cdef_replan.md) | Replan that led to Plan D |
| [cheatcode_dataset_collection.md](cheatcode_dataset_collection.md) | Operational collection recipe |
| [cheatcode_training_notes.md](cheatcode_training_notes.md) | Recorder/training implementation notes |
| [cheatcode_hackathon_success_plan.md](cheatcode_hackathon_success_plan.md) | Original long-form strategy doc; historical after later replans |
| [resume_state_2026_05_14_evening.md](resume_state_2026_05_14_evening.md) | Workstation-move resume note; historical now |

## Condensed History

- Plans A/B/C proved the model can learn gross motion, but not reliable final
  insertion. Diffusion did not fix that by itself.
- Plan D added task identity, wrist wrench, and higher image scale. It improved
  robustness to about 123, but still did not insert.
- Plan E/F added visual-servo and z-stiffness runtime assists. This improved
  proximity and produced occasional partial insertion, but not reliable full
  insertion.
- The 2026-05-15 exact-data pass showed that even near-perfect CheatCode
  demonstrations are not enough unless the training set teaches recovery from
  off-nominal final states.

The active plan is therefore not "more nominal data"; it is a filtered recovery
dataset where every kept episode is a clean full insertion after controlled
midcourse or final-approach displacement.

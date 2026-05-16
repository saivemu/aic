#!/usr/bin/env python3
"""Validate that a scored AIC collection chunk is safe to train on."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _trial_index(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return 10**9


def _load_dataset_episode_count(dataset_root: Path) -> int | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    with info_path.open() as f:
        info = json.load(f)
    return int(info["total_episodes"])


def _category_score(trial: dict[str, Any], category: str) -> float | None:
    categories = trial.get("tier_2", {}).get("categories", {})
    node = categories.get(category)
    if not isinstance(node, dict) or "score" not in node:
        return None
    return float(node["score"])


def validate(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    with args.scoring_yaml.open() as f:
        scoring = yaml.safe_load(f)

    trials = {
        name: value
        for name, value in scoring.items()
        if isinstance(name, str) and name.startswith("trial_")
    }
    dataset_episodes = _load_dataset_episode_count(args.dataset_root)

    failures: list[str] = []
    summaries: list[dict[str, Any]] = []
    if len(trials) != args.expected_trials:
        failures.append(f"expected {args.expected_trials} scored trials, got {len(trials)}")
    if dataset_episodes != args.expected_trials:
        failures.append(
            f"expected {args.expected_trials} dataset episodes, got {dataset_episodes}"
        )

    for name, trial in sorted(trials.items(), key=lambda item: _trial_index(item[0])):
        tier_1 = float(trial.get("tier_1", {}).get("score", 0.0))
        tier_2 = float(trial.get("tier_2", {}).get("score", 0.0))
        tier_3 = float(trial.get("tier_3", {}).get("score", 0.0))
        total = tier_1 + tier_2 + tier_3
        contacts = _category_score(trial, "contacts")
        insertion_force = _category_score(trial, "insertion force")
        summary = {
            "trial": name,
            "total": total,
            "tier_1": tier_1,
            "tier_2": tier_2,
            "tier_3": tier_3,
            "contacts": contacts,
            "insertion_force": insertion_force,
            "tier_3_message": trial.get("tier_3", {}).get("message", ""),
        }
        summaries.append(summary)
        if tier_3 < args.min_tier3:
            failures.append(f"{name}: tier_3 {tier_3:.3f} < {args.min_tier3:.3f}")
        if total < args.min_total:
            failures.append(f"{name}: total {total:.3f} < {args.min_total:.3f}")
        if args.require_no_contacts and contacts not in (0.0, None):
            failures.append(f"{name}: contacts score/penalty is {contacts}")
        if args.require_no_force_penalty and insertion_force not in (0.0, None):
            failures.append(f"{name}: insertion-force score/penalty is {insertion_force}")

    report = {
        "ok": not failures,
        "expected_trials": args.expected_trials,
        "dataset_episodes": dataset_episodes,
        "num_scored_trials": len(trials),
        "scoring_total": float(scoring.get("total", 0.0)),
        "min_trial_total": min((s["total"] for s in summaries), default=0.0),
        "min_tier3": min((s["tier_3"] for s in summaries), default=0.0),
        "failures": failures,
        "trials": summaries,
    }
    return not failures, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-yaml", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-trials", type=int, required=True)
    parser.add_argument("--min-total", type=float, default=90.0)
    parser.add_argument("--min-tier3", type=float, default=75.0)
    parser.add_argument("--require-no-contacts", action="store_true")
    parser.add_argument("--require-no-force-penalty", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    ok, report = validate(args)
    print(json.dumps(report, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(report, f, indent=2)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

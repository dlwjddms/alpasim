#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Parse a wizard run's results-summary.json, compute comparison metrics,
and append a row to the running experiment log.

Usage:
    python compare_variants.py record --variant-name NAME --log-dir DIR \
        --config-json '{"TRANSFUSER_CHANNEL_ORDER": "bgr", ...}' \
        [--notes "free text"]
    python compare_variants.py table [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

DEFAULT_CSV = Path(__file__).parent / "logs" / "results_log.csv"

FIELDS = [
    "variant_name",
    "log_dir",
    "config_json",
    "notes",
    "n_total",
    "n_passed",
    "n_failed",
    "n_driving_hardfail",
    "score_mean",
    "score_median",
    "score_stdev",
    "collision_at_fault_rate",
    "collision_any_rate",
    "collision_rear_rate",
    "offroad_rate",
    "wrong_lane_rate",
    "progress_rel_mean",
    "progress_clipped_rel_mean",
    "plan_deviation_mean",
    "driver_drive_rpc_duration_mean_s",
]


def load_results(log_dir: Path) -> dict:
    summary_path = log_dir / "aggregate" / "results-summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No results-summary.json under {log_dir}")
    return json.loads(summary_path.read_text())


def compute_row(variant_name: str, log_dir: Path, config: dict, notes: str) -> dict:
    """Compute comparison metrics for one run.

    IMPORTANT distinction between two different kinds of "not passed" row in
    results-summary.json:

    - Genuine driving hard-failure (collision_at_fault / offroad): the
      rollout ran to completion, every scorer computed real values, and
      `metrics`/`score_metrics` are fully populated -- `score=0.0` is a
      correct, meaningful data point and MUST be included in score_mean and
      in collision_at_fault_rate/offroad_rate, or those rates and the score
      average are silently biased upward by dropping the worst scenes.
    - True infrastructure failure (e.g. a missing local MTGS asset,
      `failure_reason` is a raw exception string, no `metrics` key at all):
      the rollout never produced a real driving outcome, so it's correctly
      excluded from driving-quality metrics -- there's nothing to measure.

    Both show up as `passed: false` in the JSON; only checking `metrics`
    presence (not the `passed`/`status` flag) tells them apart.
    """
    data = load_results(log_dir)
    rollouts = data.get("rollouts", [])
    n_total = len(rollouts)

    def has_real_metrics(r: dict) -> bool:
        # Both genuine driving hard-failures AND true infra failures (missing
        # asset, gRPC error) have a non-empty `metrics` dict -- the
        # failed-rollout path (failed_rollouts.py::failed_rollout_summary_rows)
        # populates it with {"error": ..., "rollout_id": ..., ...} instead of
        # real scorer output. The reliable signal is `score_metrics`: a
        # genuinely scored rollout (pass or driving hard-fail) always has a
        # real float for collision_at_fault; infra failures leave it None.
        score_metrics = r.get("score_metrics")
        return isinstance(score_metrics, dict) and score_metrics.get("collision_at_fault") is not None

    scored = [r for r in rollouts if has_real_metrics(r)]
    infra_failed = [r for r in rollouts if not has_real_metrics(r)]
    n_scored = len(scored)
    n_infra_failed = len(infra_failed)
    n_driving_hardfail = sum(1 for r in scored if r.get("passed") is not True)

    scores = [r["score"] for r in scored if r.get("score") is not None]

    def rate(metric_key: str) -> float | None:
        vals = []
        for r in scored:
            v = (r.get("metrics") or {}).get(metric_key)
            if v is None:
                v = (r.get("score_metrics") or {}).get(metric_key)
            if v is not None:
                vals.append(float(v))
        return (sum(1 for v in vals if v > 0) / len(vals)) if vals else None

    def mean_metric(metric_key: str) -> float | None:
        vals = []
        for r in scored:
            v = (r.get("metrics") or {}).get(metric_key)
            if v is None:
                v = (r.get("score_metrics") or {}).get(metric_key)
            if v is not None:
                vals.append(float(v))
        return statistics.mean(vals) if vals else None

    telemetry = data.get("telemetry") or {}

    row = {
        "variant_name": variant_name,
        "log_dir": str(log_dir),
        "config_json": json.dumps(config, sort_keys=True),
        "notes": notes,
        "n_total": n_total,
        "n_passed": n_scored,  # kept name for CSV/schema compat: "scored" (pass + driving hard-fail)
        "n_failed": n_infra_failed,  # kept name for CSV/schema compat: true infra failures only
        "n_driving_hardfail": n_driving_hardfail,
        "score_mean": statistics.mean(scores) if scores else None,
        "score_median": statistics.median(scores) if scores else None,
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
        "collision_at_fault_rate": rate("collision_at_fault"),
        "collision_any_rate": rate("collision_any"),
        "collision_rear_rate": rate("collision_rear"),
        "offroad_rate": rate("offroad"),
        "wrong_lane_rate": rate("wrong_lane"),
        "progress_rel_mean": mean_metric("progress_rel"),
        "progress_clipped_rel_mean": mean_metric("progress_clipped_rel"),
        "plan_deviation_mean": mean_metric("plan_deviation"),
        "driver_drive_rpc_duration_mean_s": telemetry.get(
            "driver_drive_rpc_duration_mean_s"
        ),
    }
    return row


def append_row(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_table(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"No results log yet at {csv_path}")
        return
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    cols = [
        "variant_name",
        "n_passed",
        "n_failed",
        "n_driving_hardfail",
        "score_mean",
        "score_stdev",
        "collision_at_fault_rate",
        "collision_any_rate",
        "offroad_rate",
        "wrong_lane_rate",
        "progress_rel_mean",
        "progress_clipped_rel_mean",
        "driver_drive_rpc_duration_mean_s",
    ]
    widths = {c: max(len(c), 10) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))[:12]))

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:

        def fmt(c):
            v = r.get(c, "")
            try:
                return f"{float(v):.4f}"
            except (ValueError, TypeError):
                return str(v)

        print(" | ".join(fmt(c).ljust(widths[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_record = sub.add_parser("record")
    p_record.add_argument("--variant-name", required=True)
    p_record.add_argument("--log-dir", required=True, type=Path)
    p_record.add_argument("--config-json", default="{}")
    p_record.add_argument("--notes", default="")
    p_record.add_argument("--csv", default=DEFAULT_CSV, type=Path)

    p_table = sub.add_parser("table")
    p_table.add_argument("--csv", default=DEFAULT_CSV, type=Path)

    p_rebuild = sub.add_parser(
        "rebuild",
        help="Recompute every row in an existing CSV from its log_dir/config_json "
        "(use after fixing a bug in compute_row, without re-running simulations).",
    )
    p_rebuild.add_argument("--csv", default=DEFAULT_CSV, type=Path)

    args = parser.parse_args()

    if args.command == "record":
        config = json.loads(args.config_json)
        row = compute_row(args.variant_name, args.log_dir, config, args.notes)
        append_row(row, args.csv)
        print(f"Recorded variant '{args.variant_name}':")
        for k in (
            "n_passed",
            "n_failed",
            "n_driving_hardfail",
            "score_mean",
            "collision_at_fault_rate",
            "collision_any_rate",
            "offroad_rate",
            "wrong_lane_rate",
        ):
            print(f"  {k}: {row[k]}")
    elif args.command == "table":
        print_table(args.csv)
    elif args.command == "rebuild":
        if not args.csv.exists():
            print(f"No results log at {args.csv}")
            return 1
        with args.csv.open() as f:
            old_rows = list(csv.DictReader(f))
        new_rows = []
        for old in old_rows:
            try:
                config = json.loads(old["config_json"]) if old.get("config_json") else {}
            except json.JSONDecodeError:
                config = {}
            row = compute_row(
                old["variant_name"], Path(old["log_dir"]), config, old.get("notes", "")
            )
            new_rows.append(row)
            print(
                f"{old['variant_name']}: score_mean {old.get('score_mean')} -> {row['score_mean']}"
                f"  (collision_at_fault_rate {old.get('collision_at_fault_rate')} -> {row['collision_at_fault_rate']})"
            )
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(new_rows)
        print(f"\nRebuilt {len(new_rows)} rows in {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

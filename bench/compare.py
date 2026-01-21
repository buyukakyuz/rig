#!/usr/bin/env python3
"""
Compare two benchmark results to see if an optimization helped.

Usage:
    python bench/compare.py .benchmarks/baseline.json .benchmarks/optimized.json
    python bench/compare.py --help
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_benchmark(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def format_change(baseline: float, new: float) -> str:
    if baseline == 0:
        return "N/A"

    change = ((new - baseline) / baseline) * 100

    if abs(change) < 1:
        return f"{change:+.1f}%"
    elif change > 0:
        return f"{change:+.1f}% faster"
    else:
        return f"{change:+.1f}% slower"


def format_change_inverse(baseline: float, new: float) -> str:
    if baseline == 0:
        return "N/A"

    change = ((new - baseline) / baseline) * 100

    if abs(change) < 1:
        return f"{change:+.1f}%"
    elif change < 0:
        return f"{abs(change):.1f}% faster"
    else:
        return f"{abs(change):.1f}% slower"


def compare_benchmarks(baseline_path: Path, new_path: Path) -> int:
    baseline = load_benchmark(baseline_path)
    new = load_benchmark(new_path)

    print("=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)
    print()

    def format_timestamp(ts: str) -> str:
        try:
            dt = datetime.fromtimestamp(int(ts))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return ts

    print(f"Baseline: {baseline_path.name}")
    print(f"  Model:  {baseline['metadata']['model_path']}")
    print(f"  Device: {baseline['metadata']['device']}")
    print(f"  Time:   {format_timestamp(baseline['metadata']['timestamp'])}")
    print()
    print(f"New:      {new_path.name}")
    print(f"  Model:  {new['metadata']['model_path']}")
    print(f"  Device: {new['metadata']['device']}")
    print(f"  Time:   {format_timestamp(new['metadata']['timestamp'])}")
    print()

    if baseline["metadata"]["device"] != new["metadata"]["device"]:
        print("WARNING: Different devices - comparison may not be meaningful")
        print()

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    b_summary = baseline["summary"]
    n_summary = new["summary"]

    print(f"{'Metric':<25} {'Baseline':>12} {'New':>12} {'Change':>20}")
    print("-" * 70)

    print(
        f"{'Peak Prefill (tok/s)':<25} {b_summary['peak_prefill_tok_s']:>12.1f} {n_summary['peak_prefill_tok_s']:>12.1f} {format_change(b_summary['peak_prefill_tok_s'], n_summary['peak_prefill_tok_s']):>20}"
    )
    print(
        f"{'Peak Decode (tok/s)':<25} {b_summary['peak_decode_tok_s']:>12.1f} {n_summary['peak_decode_tok_s']:>12.1f} {format_change(b_summary['peak_decode_tok_s'], n_summary['peak_decode_tok_s']):>20}"
    )
    print(
        f"{'Min TTFT (ms)':<25} {b_summary['min_ttft_ms']:>12.1f} {n_summary['min_ttft_ms']:>12.1f} {format_change_inverse(b_summary['min_ttft_ms'], n_summary['min_ttft_ms']):>20}"
    )
    print()

    print("-" * 70)
    print("PER PROMPT SIZE")
    print("-" * 70)

    baseline_by_size = {r["prompt_tokens"]: r for r in baseline["results"]}
    new_by_size = {r["prompt_tokens"]: r for r in new["results"]}

    all_sizes = sorted(set(baseline_by_size.keys()) | set(new_by_size.keys()))

    for size in all_sizes:
        b_result = baseline_by_size.get(size)
        n_result = new_by_size.get(size)

        if not b_result or not n_result:
            print(f"\nPrompt size {size}: SKIPPED (not in both benchmarks)")
            continue

        b_agg = b_result["aggregated"]
        n_agg = n_result["aggregated"]
        b_runs = len(b_result["runs"])
        n_runs = len(n_result["runs"])

        print(
            f"\nPrompt size: {size} tokens (baseline: {b_runs} runs, new: {n_runs} runs)"
        )
        print(f"  {'Metric':<23} {'Baseline':>14} {'New':>14} {'Change':>18}")
        print(f"  {'-' * 71}")

        def fmt_with_std(mean: float, std: float, runs: int) -> str:
            if runs > 1 and std > 0:
                return f"{mean:>7.1f} ±{std:>4.1f}"
            return f"{mean:>14.1f}"

        print(
            f"  {'Prefill (tok/s)':<23} {fmt_with_std(b_agg['mean_prefill_tok_s'], b_agg['std_prefill_tok_s'], b_runs):>14} {fmt_with_std(n_agg['mean_prefill_tok_s'], n_agg['std_prefill_tok_s'], n_runs):>14} {format_change(b_agg['mean_prefill_tok_s'], n_agg['mean_prefill_tok_s']):>18}"
        )
        print(
            f"  {'Decode (tok/s)':<23} {fmt_with_std(b_agg['mean_decode_tok_s'], b_agg['std_decode_tok_s'], b_runs):>14} {fmt_with_std(n_agg['mean_decode_tok_s'], n_agg['std_decode_tok_s'], n_runs):>14} {format_change(b_agg['mean_decode_tok_s'], n_agg['mean_decode_tok_s']):>18}"
        )
        print(
            f"  {'TTFT (ms)':<23} {fmt_with_std(b_agg['mean_ttft_ms'], b_agg['std_ttft_ms'], b_runs):>14} {fmt_with_std(n_agg['mean_ttft_ms'], n_agg['std_ttft_ms'], n_runs):>14} {format_change_inverse(b_agg['mean_ttft_ms'], n_agg['mean_ttft_ms']):>18}"
        )
        print(
            f"  {'Total time (ms)':<23} {fmt_with_std(b_agg['mean_total_time_ms'], 0, b_runs):>14} {fmt_with_std(n_agg['mean_total_time_ms'], 0, n_runs):>14} {format_change_inverse(b_agg['mean_total_time_ms'], n_agg['mean_total_time_ms']):>18}"
        )

    print()
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark results to evaluate optimizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python bench/compare.py .benchmarks/baseline.json .benchmarks/optimized.json
        """,
    )
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON file")
    parser.add_argument("new", type=Path, help="New benchmark JSON file to compare")

    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Error: Baseline file not found: {args.baseline}", file=sys.stderr)
        return 2

    if not args.new.exists():
        print(f"Error: New file not found: {args.new}", file=sys.stderr)
        return 2

    try:
        return compare_benchmarks(args.baseline, args.new)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 2
    except KeyError as e:
        print(f"Error: Missing field in benchmark data: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

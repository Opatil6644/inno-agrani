"""
Purpose:
    - Scans subdirectories inside a root results directory
    - Reads latency.json and throughput.json (if present)
    - Extracts:
        * avg_latency
        * latency percentiles (p10, p25, p50, p75, p90, p99)
        * throughput (images/sec)
    - Writes a consolidated CSV file
"""

import json
import csv
import argparse
from pathlib import Path


def collate_metrics(root_dir: Path, output_csv: Path) -> None:
    """
    Collate latency and throughput metrics into a single CSV file.

    Args:
        root_dir (Path): Root directory containing run subdirectories
        output_csv (Path): Output CSV file path
    """
    rows = []

    for run_dir in root_dir.iterdir():
        if not run_dir.is_dir():
            continue

        latency_file = run_dir / "latency.json"
        throughput_file = run_dir / "throughput.json"

        row = {
            "run_name": run_dir.name,
            "avg_latency": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p99": None,
            "images_per_sec": None,
        }

        # Read latency.json
        if latency_file.exists():
            with open(latency_file) as f:
                latency = json.load(f)

            row["avg_latency"] = latency.get("avg_latency")
            percentiles = latency.get("percentiles", {})
            row["p10"] = percentiles.get("10")
            row["p25"] = percentiles.get("25")
            row["p50"] = percentiles.get("50")
            row["p75"] = percentiles.get("75")
            row["p90"] = percentiles.get("90")
            row["p99"] = percentiles.get("99")

        # Read throughput.json
        if throughput_file.exists():
            with open(throughput_file) as f:
                throughput = json.load(f)

            row["images_per_sec"] = throughput.get("images/sec")

        rows.append(row)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "avg_latency",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p99",
                "images_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Latency and throughput consolidated file written to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Collate latency and throughput metrics into a CSV summary"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Root directory containing experiment/run folders",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("latency_throughput_summary.csv"),
        help="Path to output CSV file (default: latency_throughput_summary.csv)",
    )

    args = parser.parse_args()

    collate_metrics(args.results_dir, args.output_csv)


if __name__ == "__main__":
    main()
import json
import csv
from pathlib import Path
import argparse

"""
Consolidates normalized evaluation JSON results from multiple model folders
into a single aggregated CSV file.

- Updates existing model rows if results are re-run
- Inserts new rows for newly added models
"""


def consolidate_results(results_dir, results_output_csv_path):
    RESULTS_DIR = results_dir
    OUTPUT_CSV = results_output_csv_path

    # Metrics expected in the final CSV
    METRICS = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu"]
    # CSV column order
    FIELDNAMES = ["model", "precision"] + METRICS

    # Mapping between CSV metric names and JSON keys
    METRIC_KEY_MAP = {
        "ifeval": "IFEval",
        "bbh": "BBH",
        "math": "MATH",
        "gpqa": "GPQA",
        "musr": "MUSR",
        "mmlu": "MMLU"
    }

    # -----------------------------
    # Load existing CSV (if present)
    # -----------------------------
    existing_rows = {}

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row["model"]] = row


    # Flag to track whether we found at least one valid model result
    found_any_valid_results = False

    # -----------------------------
    # Scan results directory
    # -----------------------------
    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        normalized_dir = model_dir / "normalized_results"
        if not normalized_dir.exists():
            continue

        json_files = list(normalized_dir.glob("*.json"))
        if not json_files:
            continue

        # Start from existing row or initialize a new one
        row = existing_rows.get(
            model_dir.name,
            {
                "model": None,
                "precision": None,
                **{m: None for m in METRICS}
            }
        )

        # Process each JSON file for this model
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            # Skip malformed JSONs
            if "Model name" not in data:
                continue

            # Set common model-level fields
            row["model"] = data["Model name"]
            row["precision"] = data.get("Precision", row["precision"])

            # Detect and assign metric
            for metric, json_key in METRIC_KEY_MAP.items():
                if json_key in data:
                    row[metric] = data[json_key]

        # Never insert invalid rows
        if not row["model"]:
            continue

        # UPSERT (update or insert)
        existing_rows[row["model"]] = row
        found_any_valid_results = True

    # -----------------------------
    # Write aggregated CSV
    # -----------------------------

    if not found_any_valid_results:
        print("No valid model results found. CSV not generated.")
        return

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(existing_rows.values())

    print(f"Aggregated results written to: {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate normalized scores into a csv fiel")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument("output_csv_path", type=Path, help="Path to save output csv")

    args = parser.parse_args()
    results_dir = args.results_dir
    results_output_csv_path = args.output_csv_path / "aggregated_results.csv"

    consolidate_results(results_dir, results_output_csv_path)

if __name__ == "__main__":
    main()
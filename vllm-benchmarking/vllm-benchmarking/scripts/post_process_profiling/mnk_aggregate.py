"""
============================================================
Purpose:
--------
This script aggregates MNK configuration
statistics across multiple profiling runs.

What This Script Does:
--------------
1. Recursively locates all `mnk_sorted.csv` files
2. Reads each CSV into a DataFrame
3. Merges all data into a single DataFrame
4. Aggregates frequency by unique (M, N, K) combinations
5. Sorts results by frequency (highest to lowest)
6. Writes the aggregated result to an output CSV file

============================================================
"""


import pandas as pd
import glob
import os
import argparse
import sys

 
def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Aggregate MNK values across runs")
    parser.add_argument("mnk_files", help="Parent folder path containing mnk_sorted.csv files")
    parser.add_argument("results_path", help="Output CSV path for aggregated results")

    args = parser.parse_args()

    # Recursively locate all mnk_sorted.csv files
    csv_files = glob.glob(
    os.path.join(args.mnk_files, "**", "mnk_details_sorted.csv"),
    recursive=True)

    class InsufficientFilesError(Exception):
        """Raised when not enough MNK files are found for aggregation."""
    pass


    try:
        if len(csv_files) < 2:
            raise InsufficientFilesError(
                "Need more than 2 model run files to compare and aggregate, skipping mnk aggregation"
            )

    # Continue aggregation logic here
    except InsufficientFilesError as e:
        print(str(e))
        # stop execution of the script
        return

    
    frames = []
    
    # Read each file
    for f in csv_files:
        df = pd.read_csv(f)
        frames.append(df)
    
    # Combine all CSVs
    combined = pd.concat(frames, ignore_index=True)
    
    # Aggregate frequency for identical MNK
    combined_agg = (
        combined.groupby(["M", "N", "K"], as_index=False)["frequency"]
        .sum()
    )
    
    # Sort by frequency: highest to lowest
    combined_agg = combined_agg.sort_values(by="frequency", ascending=False)
    
    # Save final output
    combined_agg.to_csv(args.results_path, index=False)
    print("Successfully completed mnk aggregation")


if __name__ == "__main__":
    main()
"""
============================================================
Script Name: open_closed_kernel_classifier.py

Purpose:
--------
This script classifies GPU kernels from an Nsight Systems
kernel summary CSV file into:
1. Open-source kernels
2. Closed-source kernels

The classification is based on known open-source kernel
namespace patterns (e.g., flash::, triton, cutlass, etc.).
Writes the results into separate Excel sheets
============================================================
"""


import pandas as pd
import os
import argparse
import sys

# read kernel csv
def read_csv(filepath: str) -> pd.DataFrame:
    try:
        header_keyword = "Time (%)"

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Find header row dynamically
        header_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith(header_keyword):
                header_index = i
                break

        if header_index is None:
            raise ValueError("CSV header not found in file")

        # Read CSV starting from header
        df = pd.read_csv(filepath, skiprows=header_index)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(0)

    return df


# Main function to handle command-line arguments
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Listing open and close source kernels using kernel summary file")
    parser.add_argument("csv_filename", help="Path to the csv file containing kernel list")
    parser.add_argument("xlsx_filename", help="Path to the output csv file containing kernel_results")

    # Parse the command-line arguments
    args = parser.parse_args()

    
    filename = args.csv_filename
    output_file = args.xlsx_filename
    # Read Excel
    try:
        # Read CSV
        df = read_csv(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)


    # Pattern Matching for Open Source Kernels
    pattern = r"flash::|vllm::|at::native|std::|cub::|triton|cutlass|flashinfer::|marlin::"

    try:
        # Filter out Open Source Kernels
        open_df = df[df["Name"].str.contains(pattern, case=False, na=False)][["Name"]]

        # Filter out Closed Source Kernels(reverse of above patterns)
        closed_df = df[~df["Name"].str.contains(pattern, case=False, na=False)][["Name"]]
    except:
        print("Some issue with the kernel summary file, could not generate open/close source kernel list")
        sys.exit(1)

    data_written = False

    # Save to Excel File
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        if not closed_df.empty:
            closed_df.to_excel(writer, sheet_name="CloseSource", index=False)
            data_written = True
        else:
            closed_df.to_excel(writer, sheet_name="CloseSource", index=False)
            print("closed source kernels not found")

        if not open_df.empty:
            open_df.to_excel(writer, sheet_name="OpenSource", index=False)
            data_written = True
        else:
            open_df.to_excel(writer, sheet_name="OpenSource", index=False)
            print("open source kernels not found")


    if data_written:
        print(f"\nSaved Excel file: {output_file}")
    else:
        os.remove(output_file)
        print("\nNo kernels found. Output file deleted because it's empty.")


# Run the main function
if __name__ == "__main__":
    main()
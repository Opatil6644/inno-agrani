"""
============================================================
Script Name: nvjet_kernel_shape_extractor.py

Purpose:
--------
This script analyzes an NVIDIA Nsight Systems kernel summary
CSV file to identify and analyze `nvjet` kernels.

It extracts numeric shape patterns (e.g., 128x256x64) embedded
in kernel names, expands them into structured columns, and
produces frequency statistics for each extracted shape.
Writes expanded data and frequency statistics to a multi-sheet Excel file
============================================================
"""



import re
import pandas as pd
import argparse
import sys

# Main function to handle command-line arguments
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Listing nvjets from nsys kernel summary")
    parser.add_argument("csv_filename", help="Path to the csv file containing kernel list")
    parser.add_argument("xlsx_filename", help="Path to the output csv file containing kernel_results")

    # Parse the command-line arguments
    args = parser.parse_args()

    
    filename = args.csv_filename
    output_file = args.xlsx_filename
    # Read Excel
    try:
        # Read CSV
        df = pd.read_csv(filename,skiprows=2)
    except:
        print("File not found, skipping processing")
        sys.exit(0)


    df = df[df["Name"].str.contains("nvjet", case=False, na=False)][["Name"]]


    if df.empty:
        print("\nNo nvjet kernels found. Skipping processing.\n")
        sys.exit(0)
    
    # Extract groups
    def extract_numeric_groups(text):
        if pd.isna(text):
            return []
        groups = re.findall(r'\d+(?:x\d+)+', str(text))
        return [tuple(map(int, g.split('x'))) for g in groups]
    
    # Extract  groups
    df["numeric_groups"] = df["Name"].apply(extract_numeric_groups)
    
    # Determining Maximum number of groups in a row
    max_groups = df["numeric_groups"].apply(len).max()

    
    # Create separate columns for each group: shape1, shape2, shape3
    for i in range(max_groups):
        df[f"shape{i+1}"] = df["numeric_groups"].apply(
            lambda x: "x".join(map(str, x[i])) if i < len(x) else None
        )
    
    result_tabs = {
        "expanded_data": df
    }
    
    # Columns to check
    shape_cols = ["shape1", "shape2","shape3"]
    
    # Generate count tabs
    for col in shape_cols:
        freq = df[col].value_counts(dropna=True).reset_index()
        freq.columns = [col, "count"]
        result_tabs[f"{col}_counts"] = freq
 
    
    # Writing all sheets into one Excel file  
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, data in result_tabs.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
    


# Run the main function
if __name__ == "__main__":
    main()
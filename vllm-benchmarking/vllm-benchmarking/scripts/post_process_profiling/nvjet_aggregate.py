"""
============================================================
Purpose:
--------
This script aggregates NVJET profiling details across multiple
runs and highlights common and unique configurations.

What This Script Does:
--------------
1. Recursively locate all `nvjet_details.xlsx` files
2. Read selected sheets (sheet index 1, 2, and 3) from each file
3. Combine data across all runs per sheet
4. Count occurrences of identical configuration values
5. Write aggregated results into separate Excel sheets
6. Apply conditional formatting:
   - Green  → configurations appearing in multiple runs
   - Yellow → configurations appearing in only one run
===========================================================
"""

import pandas as pd
import glob
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import os
import argparse

# Find common and Unique rows across all dataframes
def compute_common_and_unique(df_list):
    shape = df_list.columns[0]
    x = df_list[shape].value_counts()
    result = x.reset_index()
    result.columns = [shape, 'count']
    
    return result
 

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Aggregate NVJET profiling details")
    parser.add_argument("nvjet_files", help="Parent folder path containing nvjet_details.xlsx files")
    parser.add_argument("results_path", help="Output Excel file path for aggregated results")
    args = parser.parse_args()


    # Locate all NVJET Excel files recursively
    excel_files = glob.glob(
    os.path.join(args.nvjet_files, "**", "nvjet_details.xlsx"),
    recursive=True
)
 
    class InsufficientFilesError(Exception):
        """Raised when not enough files are found for aggregation."""
    pass


    try:
        if len(excel_files) < 2:
            raise InsufficientFilesError(
                "Need more than 2 model run files to compare and aggregate, skipping nvjet aggregation"
            )

    # Continue aggregation logic here
    except InsufficientFilesError as e:
        print(str(e))
        # stop execution of the script
        return
    
    
    #Read all the sheets
    # sheet1_list = []
    sheet2_list = []
    sheet3_list = []
    sheet4_list = []
    
    for f in excel_files:
        # sheet1_list.append(pd.read_excel(f, sheet_name=0))
        sheet2_list.append(pd.read_excel(f, sheet_name=1))
        sheet3_list.append(pd.read_excel(f, sheet_name=2))
        sheet4_list.append(pd.read_excel(f, sheet_name=3))
    
    # sheet1_all_data = pd.concat(sheet1_list, ignore_index=True)
    sheet2_all_data = pd.concat(sheet2_list, ignore_index=True)
    sheet3_all_data = pd.concat(sheet3_list, ignore_index=True)
    sheet4_all_data = pd.concat(sheet4_list, ignore_index=True)


    # Compute results
    result_sheet2 = compute_common_and_unique(sheet2_all_data)
    result_sheet3 = compute_common_and_unique(sheet3_all_data)
    result_sheet4 = compute_common_and_unique(sheet4_all_data)


    # Step 1 — Write both sheets
    with pd.ExcelWriter(args.results_path, engine="openpyxl") as writer:
        result_sheet2.to_excel(writer, sheet_name="shape1", index=False)
        result_sheet3.to_excel(writer, sheet_name="shape2", index=False)
        result_sheet4.to_excel(writer, sheet_name="shape3", index=False)

    # Step 2 — Load workbook to apply styling
    wb = load_workbook(args.results_path)

    # Define color fills
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    def format_sheet(sheet):
        """Apply formatting rules to a sheet."""
        ws = wb[sheet]

        # Find the column index for 'count'
        count_col = None
        for col in range(1, ws.max_column + 1):
            if ws.cell(row=1, column=col).value == "count":
                count_col = col
                break

        if count_col is None:
            print(f"count column not found in sheet: {sheet}")
            return

        # Apply conditional formatting row by row
        for row in range(2, ws.max_row + 1):
            cell_value = ws.cell(row=row, column=count_col).value

            if cell_value is None:
                continue

            if cell_value > 1:
                fill = green_fill
            elif cell_value == 1:
                fill = yellow_fill
            else:
                continue

            # Apply fill to the entire row
            for col in range(1, ws.max_column + 1):
                ws.cell(row=row, column=col).fill = fill


    # Step 3 — Apply formatting to both sheets
    format_sheet("shape1")
    format_sheet("shape2")
    format_sheet("shape3")

    # Save the updated file
    wb.save(args.results_path)

    print(f"Nvjet aggregation and formatting applied")




if __name__ == "__main__":
    main()
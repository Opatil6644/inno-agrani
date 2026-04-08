"""
============================================================
Purpose:
--------
This script aggregates open-source and closed-source kernel
information across multiple profiling runs.

What This Script Does:
--------------
1. Recursively locates all `open_closed_source_kernels.xlsx` files
2. Reads:
   - Sheet 0: Closed-source kernels
   - Sheet 1: Open-source kernels
3. Merges data across all runs
4. Counts kernel occurrences based on the `Name` column
5. Writes aggregated results to a new Excel file with:
   - "ClosedSource" sheet
   - "OpenSource" sheet
6. Applies visual formatting:
   - Green rows → kernels appearing in multiple runs
   - Yellow rows → kernels appearing in a single run

"""

import pandas as pd
import glob
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
import os
import argparse

 
# Find common and Unique rows across all dataframes
def compute_common_and_unique(df_list):
    x = df_list['Name'].value_counts()
    result = x.reset_index()
    result.columns = ['Name', 'count']
    
    return result
 

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Aggregate open/close source kernel usage across runs")
    parser.add_argument("open_close_kernels", help="Parent directory containing open_closed_source_kernels.xlsx files")
    parser.add_argument("results_path", help="Output Excel file path for aggregated results")

    args = parser.parse_args()

    # --------------------------------------------------
    # Recursively find all kernel Excel files
    # --------------------------------------------------

    excel_files = glob.glob(
    os.path.join(args.open_close_kernels, "**", "open_closed_source_kernels.xlsx"),
    recursive=True
)

    class InsufficientFilesError(Exception):
        """Raised when not enough files are found for aggregation."""
    pass


    try:
        if len(excel_files) < 2:
            raise InsufficientFilesError(
                "Need more than 2 model run files to compare and aggregate, skipping kernel aggregation"
            )

    # Continue aggregation logic here
    except InsufficientFilesError as e:
        print(str(e))
        # stop execution of the script
        return
    

    # --------------------------------------------------
    # Read both sheets from each Excel file
    # Sheet 0 -> Closed Source
    # Sheet 1 -> Open Source
    # --------------------------------------------------
    
    #Read all the sheets
    sheet1_list = []
    sheet2_list = []
    
    for f in excel_files:
        sheet1_list.append(pd.read_excel(f, sheet_name=0))
        sheet2_list.append(pd.read_excel(f, sheet_name=1))

    # Combine all data across files
    sheet1_all_data = pd.concat(sheet1_list, ignore_index=True)
    sheet2_all_data = pd.concat(sheet2_list, ignore_index=True)


    # Compute aggregated kernel counts
    result_sheet1 = compute_common_and_unique(sheet1_all_data)

    result_sheet2 = compute_common_and_unique(sheet2_all_data)


    # Write aggregated results to Excel
    with pd.ExcelWriter(args.results_path, engine="openpyxl") as writer:
        result_sheet1.to_excel(writer, sheet_name="ClosedSource", index=False)
        result_sheet2.to_excel(writer, sheet_name="OpenSource", index=False)

    # Load workbook again to apply formatting
    wb = load_workbook(args.results_path)

    # Define color fills
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    def format_sheet(sheet):
        """
        Colors rows based on kernel frequency:
        - Green: count > 1
        - Yellow: count == 1
        """
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


    # Apply formatting to both sheets
    format_sheet("ClosedSource")
    format_sheet("OpenSource")

    # Save the updated file
    wb.save(args.results_path)

    print(f"Kernel aggregation and formatting applied")




if __name__ == "__main__":
    main()
"""
============================================================
Purpose:
--------
This script processes an MNK frequency CSV file and prepares
it for analysis by:
1. Splitting combined MNK values into separate M, N, K columns
2. Sorting MNK configurations by their frequency (descending)
3. Writing the cleaned and sorted data to a new CSV file
============================================================
"""



import pandas as pd
import argparse
import sys



# Main function to handle command-line arguments
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Sorting mnk based on frequency")
    parser.add_argument("csv_filename", help="Path to the csv file containing mnk details")
    parser.add_argument("csv_output_filename", help="Path to the sorted mnk output csv file")

    # Parse the command-line arguments
    args = parser.parse_args()

    filename = args.csv_filename
    output_file= args.csv_output_filename

    try:
        # Read CSV
        df = pd.read_csv(filename)
    except:
        print("File not found, skipping mnk sorting")
        sys.exit(0)

    if df.empty:
        print("\nFile empty, skipping mnk sorting\n")
        sys.exit(0)

    
    # Split MNK into M, N, K
    df[["M", "N", "K"]] = df["MNK"].str.split(",", expand=True)
    
    # Sort by frequency (highest to lowest)
    df = df.sort_values(by="frequency", ascending=False)
    
    # Reorder columns
    df = df[["M", "N", "K", "frequency"]]

    
    # Save output
    df.to_csv(output_file, index=False)
 
# Run the main function
if __name__ == "__main__":
    main()
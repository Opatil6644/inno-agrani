"""
============================================================
Purpose:
--------
This script parses a cuBLAS debug log file to extract:
1. Matrix dimensions (M, N, K) used in `cublasGemmEx` calls
2. Frequency of each unique MNK configuration
3. Frequency of individual cuBLAS API calls
4. Writes MNK frequency statistics to a CSV file
===========================================================
"""


import re
import argparse
import pandas as pd
 
# Function to process the log file
def extract_cublas_dimensions(log_file_path, mnk_output):
    # Regex patterns to find the cublasGemmEx and m, n, k dimensions
    cublas_pattern = re.compile(r'cublasGemmEx')
    m_pattern = re.compile(r'i!  m: type=int; val=(\d+)')
    n_pattern = re.compile(r'i!  n: type=int; val=(\d+)')
    k_pattern = re.compile(r'i!  k: type=int; val=(\d+)')
 
    m, n, k , t = None, None, None, None
    tmp_dic = {}
    max_value, mnk_v = -1, []
    max_time, prev_time, time_mnk = -1, -1, []
    # Open and read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            # Check if the line contains the cublasGemmEx function
            if cublas_pattern.search(line):
                # Found a cublasGemmEx, now look for m, n, k in subsequent lines
                m, n, k, t = None, None, None, None  # Reset m, n, k for the new block of information
                continue  # Move to the next line to find m, n, k
 
            # Now look for m, n, k on subsequent lines
            if m is None:  # Only search for m if we haven't found it yet
                m_match = m_pattern.search(line)
                if m_match:
                    m = m_match.group(1)
 
            if n is None:  # Only search for n if we haven't found it yet
                n_match = n_pattern.search(line)
                if n_match:
                    n = n_match.group(1)
 
            if k is None:  # Only search for k if we haven't found it yet
                k_match = k_pattern.search(line)
                if k_match:
                    k = k_match.group(1)
                   
            if t is None:
                t_match = re.search(r'elapsed from start [\d.]+ minutes or ([\d.]+) seconds', line)
                if t_match:
                    t = round(float(t_match.group(1)),3)
           
            # If all m, n, k and t are found, print and reset for the next cublasGemmEx block
            if m and n and k and t:
                key1 = m + "," + n + "," + k
                if key1 in tmp_dic:
                    tmp_dic[key1] += 1
                else:
                    tmp_dic[key1] = 1
                t1 = int(m)*int(n)*int(k)
                if t1 > max_value:
                    mnk_v = [m, n, k]
                    max_value = t1
                if int(t) > max_time:
                    max_time = int(t)
                    time_mnk = [m, n, k]
                m, n, k, t = None, None, None, None  # Reset for the next cublasGemmEx block

    mnk = list(tmp_dic.keys())
    freq = list(tmp_dic.values())

    if len(mnk) >= 1:
        # Write MNK frequency results to CSV
        df = pd.DataFrame({"MNK": mnk, "frequency":freq})
        df.to_csv(mnk_output, index=False)
    
    else:
        print("MNK not found, skipping mnv csv creation.")


# Extract and count cuBLAS API calls
def cublass_calls(file_path, output_csv_file):
    with open(file_path, "r+") as f:
        data = f.read()

    cudaCalls = {}

    # Split log based on cuBLAS call markers
    data = data.split("cublasStatus_t ")
    st = 0
    for item in data:
        if st == 0:
            st += 1
            continue
        item = item.split(" called:")
        if item[0] in cudaCalls:
            cudaCalls[item[0]] += 1
        else:
            cudaCalls[item[0]] = 1

    calls = list(cudaCalls.keys())
    freq = list(cudaCalls.values())
    
    df = pd.DataFrame({"Cuda cublass calls":calls, "Frequency": freq})
    df.to_csv(output_csv_file, index=False)

# Main function to handle command-line arguments
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Extract m, n, k dimensions for cublasGemmEx calls from a log file.")
    parser.add_argument("log_file", help="Path to the log file containing cublasGemmEx calls")
    parser.add_argument("mkn_output", help="Path to the save mnk output into a csv")
    parser.add_argument("cublas_func_csv", help="Path to the save the cublas function details")
 
    # Parse the command-line arguments
    args = parser.parse_args()
 
    # Extract and print the dimensions
    extract_cublas_dimensions(args.log_file, args.mkn_output)

    cublass_calls(args.log_file, args.cublas_func_csv)
 
# Run the main function
if __name__ == "__main__":
    main()
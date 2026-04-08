import json
import argparse
import sys
from pathlib import Path

"""
Split a bulk benchmark JSON file into individual files per test.
Parses an input JSON, extracts results by 'Test name', and 
saves them into a specified directory using argparse for path flexibility.
"""

def dump_results_to_files(data_list, output_dir):
    """
    Takes a list of dictionaries and saves each to a separate JSON file 
    named after its 'Test name'.
    """
    # Ensure output directory exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for entry in data_list:
        # Safely extract the test name
        test_name = entry.get('Test name')
        
        if not test_name:
            print(f"Warning: Skipping entry missing 'Test name' key: {entry}", file=sys.stderr)
            continue

        # Clean filename and build full path
        safe_filename = f"{test_name.replace(' ', '_')}.json"
        file_path = out_path / safe_filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f, indent=4)
            print(f"Successfully saved: {file_path}")
        except Exception as e:
            print(f"Error saving {safe_filename}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Split vLLM benchmark results into separate files.")
    
    # Define Path Arguments
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the source benchmark_results.json file"
    )
    parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Directory where individual JSON files will be saved"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # Validate input file existence
    if not input_path.exists():
        print(f"Error: vllm benchmark results file not found at {input_path}")
        sys.exit(1)

    # Load data
    try:
        with input_path.open("r", encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Input JSON must contain a list of results.")
            sys.exit(1)

        dump_results_to_files(data, args.output_dir)

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse input JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
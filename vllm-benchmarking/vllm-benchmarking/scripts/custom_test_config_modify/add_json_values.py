import json
import argparse
import sys
import csv
from pathlib import Path

"""
Dynamically update vLLM benchmark JSON configurations for CI/CD pipelines.
Matches a model name from CLI args against csv file to 
fetch TP size and trust_remote settings, then updates target JSON files.
"""
def get_model_details_from_csv(model_name, config_path=None):
    DEFAULT_CSV_PATH = "config/model_details/details.csv"
    """
    Searches for model details. 
    Checks config_path first (if provided); if not found, checks DEFAULT_CSV_PATH.
    """
    # Create a list of paths to check in order of priority
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    
    # Add default path if it's not already the primary path
    if Path(DEFAULT_CSV_PATH) not in search_paths:
        search_paths.append(Path(DEFAULT_CSV_PATH))

    for current_path in search_paths:
        if not current_path.exists():
            # If the user specifically provided a path that doesn't exist, we skip it
            # and try the next one (default), or print a warning.
            continue

        try:
            with open(current_path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('model', '').strip() == model_name:
                        print(f"Reading configuration from: {current_path}")
                        return {
                            "tp_size": row.get('tensor_parallel', '').strip(),
                            "trust_remote": row.get('trust_remote_code', '').strip().lower(),
                            "model_commit_id": row.get('model_commit_id', '').strip()
                        }
        except KeyError as e:
            print(f"Warning: Missing column {e} in {current_path}. Skipping...")
            continue

    # If the loop finishes without returning, the model wasn't in ANY file
    print(f"Error: Model '{model_name}' not found in {[str(p) for p in search_paths]}")
    sys.exit(1)


def update_test_config(file_path, model_name, tp_size, trust_remote, enforce_eager, compilation_mode, custom_dataset, output_len, num_prompts, model_commit_id, attention_backend):
    """Refactored logic to update a single JSON configuration file."""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: Configuration file not found at {file_path}. Skipping.")
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Custom dataset mapping dictionary
        cu_data = {"BBH":"./bbh.jsonl", "IFEVAL":"./ifeval.jsonl", "MATH": "./math.jsonl", 
                   "GPQA":"./gpqa.jsonl", "MMLU":"./mmlu.jsonl", "MUSR":"./musr.jsonl"}

        for test in data:
            # Determine if we are looking at standard parameters or server_parameters
            params_key = "server_parameters" if "server_parameters" in test else "parameters"
            
            if params_key in test:
                test[params_key]["model"] = model_name
                test[params_key]["tensor_parallel_size"] = tp_size
                test[params_key]["revision"] = model_commit_id
                test[params_key].pop("disable_log_requests", None)
                test[params_key]["attention_backend"] = attention_backend

                # Handle boolean flags: add if 'true/1', remove otherwise
                if trust_remote.lower() == '1':
                    test[params_key]["trust_remote_code"] = ""
                else:
                    test[params_key].pop("trust_remote_code", None)

                if enforce_eager.lower() == 'true':
                    test[params_key]["enforce_eager"] = ""
                else:
                    test[params_key].pop("enforce_eager", None)
                
                # Update only throughput-test json with required parameters
                if str(file_path).split('/')[-1] == "throughput-tests.json" and custom_dataset:
                    test[params_key]["dataset_name"] = "custom"
                    test[params_key]["dataset_path"] = f"{cu_data[custom_dataset]}"
                    test[params_key]["output_len"] = int(output_len)
                    test[params_key]["num_prompts"] = int(num_prompts)
                    test[params_key].pop("dataset")
                    
            # Update only serving-test json with required parameters
            if str(file_path).split('/')[-1] == "serving-tests.json" and custom_dataset:
                test["client_parameters"]["dataset_name"] = "custom"
                test["client_parameters"]["dataset_path"] = f"{cu_data[custom_dataset]}"
                test["client_parameters"]["custom_output_len"] = int(output_len)
                test["client_parameters"]["num_prompts"] = int(num_prompts)


            # Add environment variables if compilation mode is provided
            if compilation_mode and len(compilation_mode) > 3:
                test["environment_variables"] = {"VLLM_COMPILATION_MODE": compilation_mode}


        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated: {path.name}")

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Update vLLM benchmark JSONs with dynamic parameters.")
    parser.add_argument("model", help="Model name/ID")
    parser.add_argument("compilation_mode", nargs="?", default="", help="VLLM_COMPILATION_MODE value")
    parser.add_argument("enforce_eager", default="false", help="Set enforce_eager (true/false)")
    parser.add_argument("project_root", help="Path to the project root folder")
    parser.add_argument("csv_path", help="Path to model_details.csv")
    parser.add_argument("attention_backend", nargs="?", default="", help="VLLM_ATTENTION_BACKEND to use")
    parser.add_argument("dataset", nargs="?", default="", help="custom dataset name")
    parser.add_argument("num_prompts", nargs="?", default="", help="custom number of prompts")
    parser.add_argument("output_len", nargs="?", default="", help="custom output len")


    
    args = parser.parse_args()
    # 1. Fetch details from CSV
    details = get_model_details_from_csv(args.model, args.csv_path)
    
    tp_size = details['tp_size']
    trust_remote = details['trust_remote']
    model_commit_id = details['model_commit_id']
    

    print(f"Configuration found for {args.model}: TP={tp_size}, TrustRemote={trust_remote}, ModelCommitId={model_commit_id}")

    # Clean the model name in case of extra quotes from shell scripts
    model_name = args.model.strip('"')
    
    # Define relative paths to the benchmark files
    base_path = Path(args.project_root) / "vllm" / ".buildkite" / "performance-benchmarks" / "tests"
    
    targets = [
        base_path / "latency-tests.json",
        base_path / "throughput-tests.json",
        base_path / "serving-tests.json"
    ]

    for target_file in targets:
        update_test_config(
            target_file, 
            model_name, 
            tp_size, 
            trust_remote, 
            args.enforce_eager, 
            args.compilation_mode,
            args.dataset,
            args.output_len,
            args.num_prompts,
            model_commit_id,
            args.attention_backend
        )

if __name__ == "__main__":
    main()
import json
import argparse
import sys
from pathlib import Path

"""
Automate the addition of new models to vLLM benchmark suites.
Appends standardized test configurations for Latency, Serving, and Throughput 
to JSON config files while preventing duplicates and normalizing test names.
"""

# ---------------- CONFIGURATION ----------------
# Use environment variables or base paths for flexibility
BASE_CFG_PATH = Path("config/test_configurations/cuda")

TEST_SUITES = {
    "latency": {
        "file": BASE_CFG_PATH / "latency-tests.json",
        "param_key": "parameters",
        "template": lambda model, name: {
            "test_name": f"latency_{name}_tp1",
            "parameters": {
                "model": model,
                "tensor_parallel_size": 1,
                "load_format": "dummy",
                "num_iters_warmup": 5,
                "num_iters": 15
            }
        }
    },
    "serving": {
        "file": BASE_CFG_PATH / "serving-tests.json",
        "param_key": "server_parameters",
        "template": lambda model, name: {
            "test_name": f"serving_{name}_tp1_sharegpt",
            "qps_list": [1, 4, 16, "inf"],
            "server_parameters": {
                "model": model,
                "tensor_parallel_size": 1,
                "swap_space": 16,
                "disable_log_stats": "",
                "disable_log_requests": "",
                "load_format": "dummy"
            },
            "client_parameters": {
                "model": model,
                "backend": "vllm",
                "dataset_name": "sharegpt",
                "dataset_path": "./ShareGPT_V3_unfiltered_cleaned_split.json",
                "num_prompts": 200
            }
        }
    },
    "throughput": {
        "file": BASE_CFG_PATH / "throughput-tests.json",
        "param_key": "parameters",
        "template": lambda model, name: {
            "test_name": f"throughput_{name}_tp1",
            "parameters": {
                "model": model,
                "tensor_parallel_size": 1,
                "load_format": "dummy",
                "dataset": "./ShareGPT_V3_unfiltered_cleaned_split.json",
                "num_prompts": 200,
                "backend": "vllm"
            }
        }
    }
}

def normalize_name(model_id: str) -> str:
    """Creates a clean slug for test names."""
    return (
        model_id.split("/")[-1]
        .replace(".", "")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )

def process_suite(suite_name, model_id):
    cfg = TEST_SUITES[suite_name]
    file_path = cfg["file"]
    
    # Load existing data
    if not file_path.exists():
        print(f"Creating new file for {suite_name}...")
        data = []
    else:
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return "Error"

    # Check for duplicates
    pk = cfg["param_key"]
    existing_models = {item[pk]["model"] for item in data if pk in item}
    
    if model_id in existing_models:
        return "Skipped"

    # Add new entry
    clean_name = normalize_name(model_id)
    new_entry = cfg["template"](model_id, clean_name)
    data.append(new_entry)

    # Save back
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return "Added"
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return "Error"

def main():
    parser = argparse.ArgumentParser(description="Onboard a new model to benchmark suites.")
    parser.add_argument("model_name", help="The full HuggingFace model ID")
    args = parser.parse_args()
    
    model_id = args.model_name.strip('"')
    print(f"Processing Model: {model_id}\n" + "-"*30)

    summary = {}
    for suite in TEST_SUITES:
        result = process_suite(suite, model_id)
        summary[suite.upper()] = result

    for suite, status in summary.items():
        print(f"{suite:12}: {status}")

if __name__ == "__main__":
    main()
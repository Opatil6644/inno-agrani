"""
Automate the large-scale evaluation of Large Language Models (LLMs) using the 'lm-evaluation-harness'.
The script reads model configurations from a YAML file, executes evaluations, organizes raw 
results into a structured directory tree, and triggers a post-processing normalization script.

1. Argument Parsing: Supports metrics, output paths, and custom config file locations.
2. Configuration Management: Dynamically loads models and their types (base/instruct).
3. Directory Automation: Automatically creates nested folder structures for raw and processed data.
4. Post-Processing: Automatically passes the latest results to 'score_normalize.py'.
5. Logging & Error Handling: Provides a detailed audit trail of successes and failures.
"""

import yaml
import subprocess
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path


def run_command(cmd, name):
    """
    Executes a shell command via subprocess.
    Args:
        cmd (list): The list of command-line arguments.
        name (str): A friendly name for the process (for logging).
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        logging.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed for {name} with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        logging.error(f"Executable not found: {e}")
        return False

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="LLM Evaluation Harness Runner")
    parser.add_argument("metrics", help="The evaluation metrics/tasks (e.g., leaderboard_bbh)")
    parser.add_argument("results_path", help="Base directory to store results")
    parser.add_argument(
        "--config", 
        default="models.yaml", 
        help="Path to the models YAML file (default: models.yaml)"
    )
    parser.add_argument(
    "--trust-remote-code",
    action="store_true",
    help="Allow executing remote code from Hugging Face Hub (required for some models)"
    )

    
    args = parser.parse_args()

    # Convert to Path objects for easier manipulation
    metrics = args.metrics
    base_results_path = Path(args.results_path)
    models_file = Path(args.config)
    normalize_script = Path("evaluate/score_normalize.py")
    consolidate_script = Path("evaluate/consolidate_results.py")
    trust_remote_code = args.trust_remote_code

    # Define log file (Daily format: eval_session_YYYYMMDD.log)
    log_file_path = base_results_path / f"eval_session_{datetime.now().strftime('%Y%m%d')}.log"

    # Configure Logging to capture INFO and higher, sending to both File and Console
    # mode='a' ensures we keep a history of all runs for that day
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*60)
    logging.info("NEW EVALUATION SESSION STARTED")
    logging.info(f"Metrics: {args.metrics}")
    logging.info(f"Results Path: {base_results_path.absolute()}")
    logging.info("="*60)

    # --- Validation ---
    if not models_file.exists():
        logging.error(f"Configuration file not found at: {models_file.absolute()}")
        sys.exit(1)
    
    logging.info(f"Using config file: {models_file.absolute()}")

    # --- Load Configuration ---
    try:
        with open(models_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        sys.exit(1)
    
    models = config.get('models', [])
    if not models:
        logging.error("No models found in the configuration file.")
        sys.exit(1)

    success_count = 0
    failed_count = 0

    # --- Execution Loop ---
    for idx, model in enumerate(models, 1):
        name = model.get('name')
        model_type = model.get('type')
        
        # Guard clause for malformed YAML entries
        if not name or not model_type:
            logging.warning(f"Skipping index {idx}: Incomplete config (name or type missing)")
            continue

        logging.info(f"[{idx}/{len(models)}] Starting evaluation: {name}")

        # Path Management
        # Organize folder structure: Results -> Model_Name -> (raw_results / normalized_results)
        safe_name = name.replace('/', '_').replace(' ', '_')
        model_root = base_results_path / "Results" / safe_name
        raw_path = model_root / "raw_results"
        norm_path = model_root / "normalized_results"

        # Ensure directories exist before running
        raw_path.mkdir(parents=True, exist_ok=True)
        norm_path.mkdir(parents=True, exist_ok=True)

        # Generate unique filename using timestamp to avoid overwriting previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = raw_path / f"{timestamp}_{metrics}_results.json"

        # --- STEP 1: LM-EVALUATION-HARNESS EXECUTION ---
        eval_cmd = [
            'lm-eval',
            '--model', 'hf', # HuggingFace model type
            f'--model_args=pretrained={name},dtype=auto',
            f'--tasks={metrics}',
            '--batch_size=auto',
            f'--output_path={str(result_file)}'
        ]
        
        # Add specific flags if the model is an 'instruct' or 'chat' tuned model
        if model_type == 'instruct':
            eval_cmd.extend(['--apply_chat_template', '--fewshot_as_multiturn'])

        # Conditionally allow remote code
        if trust_remote_code:
            eval_cmd.append('--trust_remote_code')

        # Run evaluation; proceed to normalization only if evaluation succeeds
        if run_command(eval_cmd, name):
            results_dir = Path(raw_path)
            latest_file = max(results_dir.glob(f"*{metrics}_results*.json"))
            print("latest file", latest_file)

            # --- STEP 2: SCORE NORMALIZATION ---
            if latest_file.exists():
                norm_cmd = [
                    sys.executable,
                    str(normalize_script),
                    str(latest_file),
                    str(norm_path),
                    metrics,
                    name
                ]
                if run_command(norm_cmd, f"{name}_normalization"):
                    success_count += 1
                else:
                    failed_count += 1
            else:
                logging.error(f"Result file {result_file} was not found after evaluation.")
                failed_count += 1
        else:
            failed_count += 1


    # Consolidate normalized results into a csv file
    con_cmd = [     sys.executable,
                    str(consolidate_script),
                    str(Path(args.results_path + "/Results")),
                    str(Path(args.results_path + "/Results")),]

    run_command(con_cmd, "Consolidating normalized results")

    # --- Final Summary ---
    logging.info("=" * 30)
    logging.info("EVALUATION COMPLETE")
    logging.info(f"Total processed: {len(models)}")
    logging.info(f"Success: {success_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info("=" * 30)

if __name__ == "__main__":
    main()
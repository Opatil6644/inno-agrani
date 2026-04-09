#!/bin/bash

# ==============================================================================
# PURPOSE: Master ASR Leaderboard Orchestrator
# ------------------------------------------------------------------------------
# This script automates the evaluation of multiple ASR models (NVIDIA NeMo 
# Canary/Parakeet and HuggingFace Transformers/Whisper) across a list of datasets.
#
# Key Functions:
#   1. Loads environment variables (HF_TOKEN) from a .env file.
#   2. Validates existence of model and dataset configuration files.
#   3. Routes models to their specific execution environments (nemo_asr vs. 
#      transformers) based on naming conventions.
#   4. Executes specialized sub-scripts for each model-dataset pair.
#   5. Aggregates all raw results into a final Excel leaderboard summary.
# ==============================================================================

# --- 1. SET PATHS ---
ROOT_DIR=$(pwd)
MODEL_FILE="$ROOT_DIR/models.txt"
DATASET_FILE="$ROOT_DIR/datasets.txt"
RESULTS_DIR="$ROOT_DIR/Results"

# Add ROOT to python path so sub-scripts can find shared modules like 'normalizer'
export PYTHONPATH="$PYTHONPATH:$ROOT_DIR"

# --- 0. LOAD ENVIRONMENT VARIABLES ---
if [ -f "../env" ]; then
    # Export variables from .env, ignoring comments and empty lines
    export $(grep -v '^#' "../env" | xargs)
    echo "Loaded environment variables from .env"
else
    echo "Warning: .env file not found"
fi

# Now $HF_TOKEN is available for the rest of the script

# Ensure Results directory exists
mkdir -p "$RESULTS_DIR"

# --- 2. VALIDATION ---
if [[ ! -f "$MODEL_FILE" ]]; then
    echo " Error: models.txt not found at $MODEL_FILE"
    exit 1
fi

if [[ ! -f "$DATASET_FILE" ]]; then
    echo " Error: datasets.txt not found at $DATASET_FILE"
    exit 1
fi

# --- 3. READ INPUTS ---
# Read models, skipping empty lines and comments
mapfile -t MODELS < <(grep -vE '^(\s*#|\s*$)' "$MODEL_FILE")
# Read datasets, skipping empty lines and comments
mapfile -t DATASETS < <(grep -vE '^(\s*#|\s*$)' "$DATASET_FILE")

echo " Starting Master Leaderboard Run"
echo "Found ${#MODELS[@]} models and ${#DATASETS[@]} datasets."

# --- 4. MAIN LOOP ---
for MODEL_ID in "${MODELS[@]}"; do
    echo -e "\n\033[1;34m====================================================\033[0m"
    echo -e "\033[1;34mTARGET MODEL: $MODEL_ID\033[0m"
    echo -e "\033[1;34m====================================================\033[0m"

    # A. ROUTING LOGIC: Determine which folder and shell script to use
    if [[ "$MODEL_ID" == *"canary"* ]]; then
        TARGET_DIR="nemo_asr"
        EXEC_SCRIPT="run_salm_updated.sh"
   
    elif [[ "$MODEL_ID" == *"parakeet"* ]]; then
        TARGET_DIR="nemo_asr"
        EXEC_SCRIPT="run_parakeet_updated.sh"

    elif [[ "$MODEL_ID" == *"whisper"* ]]; then
            TARGET_DIR="transformers"
            EXEC_SCRIPT="run_whisper_updated.sh"
    fi

    echo " Routing to directory: $TARGET_DIR"

    # B. EXECUTION
    # Enter the folder, execute the model's specific script, and return
    if [[ -d "$ROOT_DIR/$TARGET_DIR" ]]; then
        cd "$ROOT_DIR/$TARGET_DIR" || exit
        
        # We pass MODEL_ID as an argument to the sub-script
        # We also pass the absolute path to DATASETS_FILE
        bash "$EXEC_SCRIPT" "$MODEL_ID" "$DATASET_FILE"
        
        cd "$ROOT_DIR" || exit
    else
        echo " Warning: Directory $TARGET_DIR does not exist. Skipping $MODEL_ID."
    fi
done

# --- 5. POST-PROCESSING ---
echo -e "\n\033[1;32m====================================================\033[0m"
echo -e "\033[1;32m RUNNING GLOBAL AGGREGATION\033[0m"
echo -e "\033[1;32m====================================================\033[0m"

# # Call the centralized post-processing script
# if [[ -d "$ROOT_DIR/post_processing_scripts" ]]; then
#     python "$ROOT_DIR/post_processing_scripts/nvjet_aggregate.py" \
#         "$RESULTS_DIR" \
#         "$RESULTS_DIR/final_leaderboard_summary.xlsx"
# else
#     echo " Warning: post_processing_scripts folder not found."
# fi

echo -e "\n Master Run Complete. Results are in $RESULTS_DIR"

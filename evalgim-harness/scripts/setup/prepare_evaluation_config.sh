#!/bin/bash
###############################################################################
#
# Purpose:
#   - Read user configuration from user_details.txt
#   - Pass all required parameters into update_config.py
#   - Prepare configuration required for the main evaluation pipeline
################################################################################

# Exit immediately if a command exits with a non-zero status
set -e  

############### Helper function: pretty status message ###############
print_status_message() {
    local text="$1"
    local border="================================================================================="
    echo ""
    echo "$border"
    echo "$text"
    echo "$border"
    echo ""
}


# Absolute path of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# User input details file
CONFIG_FILE="$SCRIPT_DIR/../../config/user_evaluation_settings.json"

# Eval input json file
CONFIG_PATH="$SCRIPT_DIR/../../config/config.json"

# Function to read a key's value from the config file
get_value() {
    local key="$1"
    local value

    value=$(jq -r ".${key}" "$CONFIG_FILE")

    if [[ "$value" == "null" || -z "$value" ]]; then
        echo "ERROR: Missing required key: $key"
        exit 1
    fi

    echo "$value"
}

# Read each variable safely
HF_TOKEN=$(get_value "hf_token")
COCO_DATA_PATH=$(get_value "coco_data_path")
RESULTS_PATH=$(get_value "results_path")
NUM_SAMPLES=$(get_value "num_samples")
BATCH_SIZE=$(get_value "batch_size")
SAVE_GEN_IMG=$(get_value "save_generated_images")


MESSAGE="Updating config file.................."
print_status_message "$MESSAGE"
python3 ./scripts/setup/update_config.py --hf_token $HF_TOKEN --coco_dataset_path $COCO_DATA_PATH --results_path $RESULTS_PATH --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --save_gen_img $SAVE_GEN_IMG --config_json_path $CONFIG_PATH
MESSAGE="Updating config file completed"





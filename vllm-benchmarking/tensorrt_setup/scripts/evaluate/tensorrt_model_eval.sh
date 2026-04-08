#!/bin/bash

#################################################################################
# PURPOSE: Performance & Kernel Profiling Runner
# -------------------------------------------------------------------------------
# This script orchestrates a high-level benchmarking suite for hf models. 
# It performs the following:
# 
# 1. ORCHESTRATION: Parses 'master_model_list.txt', 'backend_list.txt' and 'user_eval_setting.json' to manage 
#    execution across multiple LLM architectures.
# 2. DEEP PROFILING: Wraps the benchmark in 'nsys profile' to capture hardware-
#    level CUDA kernels and cuBLAS GEMM operations (MNK dimensions).
# 3. POST-PROCESSING: Aggregates performance JSONs and kernel summaries into 
#    comparative Excel/CSV reports for cross-model analysis.
#################################################################################


# Exit immediately if a command exits with a non-zero status
set -e  

# This ensures any internal HF calls use the token correctly
#huggingface-cli login --token "$HF_TOKEN"
############### Helper: Check config.json exists ###############
CONFIG_JSON_PATH="config/user_eval_setting.json"

if [[ ! -f "$CONFIG_JSON_PATH" ]]; then
  echo "Config file not found at: $CONFIG_JSON_PATH"
  exit 1
fi

############### Helper function: pretty box ###############
print_box() {
    local text="$1"
    local border="================================================================================="
    echo ""
    echo "$border"
    echo "Inferencing started for MODEL: $text"
    echo "$border"
    echo ""
}

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

MESSAGE="Loading model names from master list"
print_status_message "$MESSAGE"


# Use this filepath if users hasn't passed the argument
DEFAULT_MODEL_LIST="master_model_list.txt"

# Check if user has passed arugment 
if [[ $# -ge 1 ]]; then
    MASTER_MODEL_LIST_PATH="$1"
    MESSAGE="Reading model list from user-provided path: $MASTER_MODEL_LIST_PATH"
else
    MASTER_MODEL_LIST_PATH="$DEFAULT_MODEL_LIST"
    MESSAGE="No model list argument provided. Reading from default path"
fi

print_status_message "$MESSAGE"

# Load model names:
# - Ignore lines starting with #
# - Ignore empty lines
# readarray -t MODEL_NAMES_ARRAY < <(grep -vE '^\s*#' master_model_list.txt | sed '/^\s*$/d')
readarray -t MODEL_NAMES_ARRAY < <(grep -vE '^\s*#' "$MASTER_MODEL_LIST_PATH" | sed '/^\s*$/d')


############### Validate model list ###############
# Validate: at least one model must be present
if [[ ${#MODEL_NAMES_ARRAY[@]} -eq 0 ]]; then
    MESSAGE="ERROR: No models selected. Uncomment at least one model in master_model_list.txt"
    print_status_message "$MESSAGE"
    exit 1
fi

# Show loaded models for debugging
MESSAGE="Models selected"
print_status_message "$MESSAGE"
printf '%s\n' "${MODEL_NAMES_ARRAY[@]}"
echo ""

############### Load config.json values ###############

# Default results path (used if not provided in config.json)
# -------------------------------
DEFAULT_RESULTS_PATH="Results"

# -------------------------------
# Read results_path from config.json
# - `// empty` avoids returning "null"
# -------------------------------
RESULTS_PATH=$(jq -r '.results_path // empty' "$CONFIG_JSON_PATH")

# Huggingface Login
python scripts/setup/hf_login.py --config_json_path $CONFIG_JSON_PATH

# -------------------------------
# If results_path is empty, use default
# -------------------------------
if [ -z "$RESULTS_PATH" ]; then
    RESULTS_PATH="$DEFAULT_RESULTS_PATH"
    echo "results_path not provided. Using default path: $RESULTS_PATH"
else
    echo "Using results_path from config.json: $RESULTS_PATH"
fi

# Open/close source custom python filename
KERNEL_LISTING_FILE="scripts/post_processing_scripts/open_closed_kernel_listing.py"

#read all sorted mnk and aggregate into one
MNK_AGGREGATE="scripts/post_processing_scripts/mnk_aggregate.py"

#read all open/closed source kernels and aggregate into one
KERNEL_AGGREGATE="scripts/post_processing_scripts/kernel_aggregate.py"

#aggregates all nvjet details into one
NVJET_AGGREGATE="scripts/post_processing_scripts/nvjet_aggregate.py"

# Filename to store mnk aggregation details
MNK_AGGREGATE_FILENAME=$RESULTS_PATH/Results/"mnk_aggregate.csv"

# Filename to store open/close source kernel aggregation details
OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME=$RESULTS_PATH/Results/"open_close_source_kernels_aggregate.xlsx"

# Filename to store nvjet aggregation details
NVJET_AGGREGATE_FILENAME=$RESULTS_PATH/Results/"nvjet_aggregate.xlsx"


# Mnk frequency script
MNK_FREQ_SCRIPT="scripts/post_processing_scripts/check_mnk.py"

# Listing nvjets along with shape
NVJET_LIST="scripts/post_processing_scripts/nvjet_list.py"

# Sorting mnk based on frequency filename
MNK_SORT="scripts/post_processing_scripts/mnk_sort.py"


MESSAGE="Checking image path"
print_status_message "$MESSAGE"


# Define the backend list file
INPUT_FILE="backend_list.txt"

# 1. Check if the file exists before proceeding
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: File '$INPUT_FILE' not found." >&2
    exit 1
fi

readarray -t BACKENDS < <(grep -vE '^\s*#' "$INPUT_FILE" | sed '/^\s*$/d')


for MODEL_NAME in "${MODEL_NAMES_ARRAY[@]}"; do
  # MODEL_NAME usually contains quotes → strip them
  MODEL_NAME=$(echo $MODEL_NAME | tr -d '"')
  print_box "$MODEL_NAME"

  # Replace / or \ with -- so it becomes a safe folder name
  PRE_PROCESSED_MODEL_NAME=${MODEL_NAME//[\/\\]/--}
  SAFE_MODEL_NAME=$(basename "$MODEL_NAME")
  echo $SAFE_MODEL_NAME

  #ENGINE_PATH="/models_test/${SAFE_MODEL_NAME}_engine"
  #MODEL_FILE_PATH="/models_test/${SAFE_MODEL_NAME}"
  MODEL_FILE_PATH="./${SAFE_MODEL_NAME}"


    
  for BACKEND in "${BACKENDS[@]}"; do
      # Skip empty lines
      [[ -z "$BACKEND" ]] && continue

    echo "Processing compiler: $BACKEND"


    # Folder path to profiling reports
    MODEL_PROF_RESULTS_PATH="$RESULTS_PATH/Results/$PRE_PROCESSED_MODEL_NAME/Profile_Details/$BACKEND"

    # Check if folder already exists else skip folder creation
    if [ ! -d "$MODEL_PROF_RESULTS_PATH" ]; then
        MESSAGE="Profiling folder does not exist. Creating to store profiling details"
        print_status_message "$MESSAGE"
        mkdir -p "$MODEL_PROF_RESULTS_PATH"
    else
        MESSAGE="Profiling folder already exists. Skipping creation"
        print_status_message "$MESSAGE"
    fi


    # Nsys report filename
    REPORT_FILENAME="model_nsys_report"
    # Kernel details filename
    KERNEL_SUMMARY_FILENAME="kernel_summary.csv"

    # Path to store cublas logs
    CUBLAS_FILE_PATH=$MODEL_PROF_RESULTS_PATH/"cublas_log.txt"

    # Path to store cublas function list
    CUBLAS_FUNC_OUTPUT=$MODEL_PROF_RESULTS_PATH/"cublas_func.csv"

    # Path to store m,n,k details
    MNK_OUTPUT=$MODEL_PROF_RESULTS_PATH/"mnk.csv"

    # Path to store sorted m,n,k details
    MNK_SORTED_OUTPUT=$MODEL_PROF_RESULTS_PATH/"mnk_sorted.csv"

    # Path to save nvjet details
    KERNEL_RESULTS=$MODEL_PROF_RESULTS_PATH/"nvjet_details.xlsx"

    # Check and delete existing cuBLAS log file
    if [ -f "$CUBLAS_FILE_PATH" ]; then
        rm -f "$CUBLAS_FILE_PATH"
        MESSAGE="Existing cuBLAS log file deleted"
        print_status_message "$MESSAGE"
    else
        MESSAGE="No existing cuBLAS log file found. Proceeding"
        print_status_message "$MESSAGE"
    fi


    # Environment variables set to capture cublas logs
    export CUBLAS_LOGINFO_DBG=1
    export CUBLAS_LOGDEST_DBG=$CUBLAS_FILE_PATH

    # Check and delete existing .nsys-rep and .sqlite files
    if [ -d "$MODEL_PROF_RESULTS_PATH" ]; then

        # Find and delete .nsys-rep and .sqlite files if they exist
        find "$MODEL_PROF_RESULTS_PATH" -type f \( -name "*.nsys-rep" -o -name "*.sqlite" \) -exec rm -f {} +

        MESSAGE="Existing .nsys-rep and .sqlite files deleted (if present)"
        print_status_message "$MESSAGE"

    else
        MESSAGE="Profiling results directory not found. Skipping cleanup"
        print_status_message "$MESSAGE"
    fi
    

    MESSAGE="Model inference and profiling started"
    print_status_message "$MESSAGE"
    nsys profile \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output="$MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME" \
    --force-overwrite=true \
    python scripts/evaluate/inference.py --model_name $MODEL_NAME --backend $BACKEND --output_path $MODEL_PROF_RESULTS_PATH
    # Converts the kernel_summary report into a csv file
    nsys stats --force-export true --report cuda_gpu_kern_sum --format csv $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.nsys-rep > $MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME

    # Custom python file to list out open/close source kernels
    python $KERNEL_LISTING_FILE $MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME $MODEL_PROF_RESULTS_PATH/open_closed_source_kernels.xlsx

    python $MNK_FREQ_SCRIPT $CUBLAS_FILE_PATH $MNK_OUTPUT $CUBLAS_FUNC_OUTPUT
    python $NVJET_LIST $MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME $KERNEL_RESULTS
    python $MNK_SORT $MNK_OUTPUT $MNK_SORTED_OUTPUT
    
    MESSAGE="MNK frequency + nvjet listing + mnk sorting completed"
    print_status_message "$MESSAGE"

    # Remove the files that are not required
    rm -rf $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.nsys-rep
    rm -rf $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.sqlite

    # Delete cuBLAS log file
    if [ -f "$CUBLAS_FILE_PATH" ]; then
        rm -f "$CUBLAS_FILE_PATH"
        MESSAGE="Cublas log file deleted"
        print_status_message "$MESSAGE"
    else
        MESSAGE="No Cublas log file found"
        print_status_message "$MESSAGE"
    fi

    MESSAGE="Inference + Profiling completed!"
    print_status_message "$MESSAGE"

  done < "$INPUT_FILE"
  

done


python $MNK_AGGREGATE $RESULTS_PATH $MNK_AGGREGATE_FILENAME
python $KERNEL_AGGREGATE $RESULTS_PATH $OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME
python $NVJET_AGGREGATE $RESULTS_PATH $NVJET_AGGREGATE_FILENAME

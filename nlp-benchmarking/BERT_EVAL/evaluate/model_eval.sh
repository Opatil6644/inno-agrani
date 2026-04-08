#!/bin/bash

#################################################################################
# PURPOSE: Automated Bert Evaluation, Profiling and Kernel Analysis Pipeline
# -------------------------------------------------------------------------------
# This script automates the systematic profiling of multiple models (from a master
# list) using NVIDIA Nsight Systems. Key functionalities include:
# 
# 1. CONFIGURATION: Parses 'user_eval_setting.json' for paths and authenticates with 
#    HuggingFace for model access.
# 2. PROFILING: Orchestrates 'nsys profile' to trace CUDA, NVTX, and cuBLAS 
#    activities during model evaluation.
# 3. KERNEL TELEMETRY: Captures cuBLAS debug logs to extract GEMM parameters 
#    (M, N, K dimensions) and calculates kernel frequency/distribution.
# 4. CLASSIFICATION: Categorizes GPU kernels into open-source vs. closed-source 
#    (proprietary) variants for architectural analysis.
# 5. AGGREGATION: Compiles individual model metrics into global reports 
#    (MNK aggregate, NVJET details, and kernel summaries) in CSV/XLSX formats.
# 6. CLEANUP: Automatically purges high-volume artifacts like .nsys-rep and 
#    .sqlite files to maintain storage efficiency.
#################################################################################

# Exit immediately if a command exits with a non-zero status
set -e  

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
    echo "Evaluation started for MODEL: $text"
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

############### Load config.json values ###############

# Default results path (used if not provided in config.json)
# -------------------------------
DEFAULT_RESULTS_PATH="Results"

# -------------------------------
# Read results_path from config.json
# - `// empty` avoids returning "null"
# -------------------------------
RESULTS_PATH=$(jq -r '.results_path // empty' "$CONFIG_JSON_PATH")

python evaluate/hf_login.py --config_json_path $CONFIG_JSON_PATH


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
KERNEL_LISTING_FILE="post_processing_scripts/open_closed_kernel_listing.py"

#read all sorted mnk and aggregate into one
MNK_AGGREGATE="post_processing_scripts/mnk_aggregate.py"

#read all open/closed source kernels and aggregate into one
KERNEL_AGGREGATE="post_processing_scripts/kernel_aggregate.py"

#aggregates all nvjet details into one
NVJET_AGGREGATE="post_processing_scripts/nvjet_aggregate.py"

# Filename to store mnk aggregation details
MNK_AGGREGATE_FILENAME=$RESULTS_PATH/Results/"mnk_aggregate.csv"

# Filename to store open/close source kernel aggregation details
OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME=$RESULTS_PATH/Results/"open_close_source_kernels_aggregate.xlsx"

# Filename to store nvjet aggregation details
NVJET_AGGREGATE_FILENAME=$RESULTS_PATH/Results/"nvjet_aggregate.xlsx"


# Mnk frequency script
MNK_FREQ_SCRIPT="post_processing_scripts/check_mnk.py"

# Listing nvjets along with shape
NVJET_LIST="post_processing_scripts/nvjet_list.py"

# Sorting mnk based on frequency filename
MNK_SORT="post_processing_scripts/mnk_sort.py"


# Default model details
MODEL_CONFIG="models.yaml"

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "Error: yq is not installed."
    exit 1
fi


# If the file is empty or 'models' key doesn't exist, yq returns 0 or null
NUM_MODELS=$(yq -r '.models | length // 0' "$MODEL_CONFIG")

# --- NEW CONDITION: Check if at least one model is selected ---
if [[ "$NUM_MODELS" -eq 0 || "$NUM_MODELS" == "null" ]]; then
    echo "------------------------------------------------"
    echo "ERROR: No enabled models found in $MODEL_CONFIG."
    echo "Please uncomment at least one model to proceed."
    echo "------------------------------------------------"
    exit 1
fi


echo -e "${YELLOW}Starting benchmarks for ${NUM_MODELS} model(s)...${NC}"
echo ""





for ((i=0; i<$NUM_MODELS; i++)); do
  MODEL=$(yq -r ".models[$i].name" "$MODEL_CONFIG")
  TYPE=$(yq -r ".models[$i].type" "$MODEL_CONFIG")
  # MODEL_NAME usually contains quotes → strip them
  MODEL_NAME=$(echo $MODEL | tr -d '"')

  print_box "$MODEL_NAME"

  # Replace / or \ with -- so it becomes a safe folder name
  PRE_PROCESSED_MODEL_NAME=${MODEL_NAME//[\/\\]/--}

  # Folder path to profiling reports
  MODEL_PROF_RESULTS_PATH="$RESULTS_PATH/Results/$PRE_PROCESSED_MODEL_NAME/Profile_Details"

  # Folder path to store performance results
  MODEL_PERF_RESULTS_PATH="$RESULTS_PATH/Results/$PRE_PROCESSED_MODEL_NAME/Eval_Details"

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
  

  MESSAGE="Model Evaluation and profiling started"
  print_status_message "$MESSAGE"

  nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --force-overwrite true --output $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME  python evaluate/model_eval.py \
  --model_name $MODEL_NAME \
  --type $TYPE \
  --output_path $MODEL_PERF_RESULTS_PATH

  # Converts the kernel_summary report into a csv file
  nsys stats --force-export true --report cuda_gpu_kern_sum --format csv $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.nsys-rep > $MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME

  python $MNK_FREQ_SCRIPT $CUBLAS_FILE_PATH $MNK_OUTPUT $CUBLAS_FUNC_OUTPUT
  python $MNK_SORT $MNK_OUTPUT $MNK_SORTED_OUTPUT
  
  # Custom python file to list out open/close source kernels
  python $KERNEL_LISTING_FILE $MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME $MODEL_PROF_RESULTS_PATH/open_closed_source_kernels.xlsx
  python $NVJET_LIST $MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME $KERNEL_RESULTS

  
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

  MESSAGE="Evaluation + Profiling completed!"
  print_status_message "$MESSAGE"


done


python $MNK_AGGREGATE $RESULTS_PATH $MNK_AGGREGATE_FILENAME
python $KERNEL_AGGREGATE $RESULTS_PATH $OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME
python $NVJET_AGGREGATE $RESULTS_PATH $NVJET_AGGREGATE_FILENAME
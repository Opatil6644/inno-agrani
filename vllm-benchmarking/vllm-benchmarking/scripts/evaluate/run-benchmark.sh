#!/bin/bash

#################################################################################
# PURPOSE: vLLM Performance & Kernel Profiling Runner
# -------------------------------------------------------------------------------
# This script orchestrates a high-level benchmarking suite for vLLM models 
# within local environment. It performs the following:
# 
# 1. ORCHESTRATION: Parses 'models.yaml', 'dynamic_config' and 'static_config.json' to manage 
#    batch execution across multiple LLM architectures.
# 2. DYNAMIC CONFIGURATION: Modifies vLLM nightly benchmark tests on-the-fly 
#    (compilation modes, eager execution, and attention backends).
# 3. DEEP PROFILING: Wraps the benchmark in 'nsys profile' to capture hardware-
#    level CUDA kernels and cuBLAS GEMM operations (MNK dimensions).
# 4. POST-PROCESSING: Aggregates performance JSONs and kernel summaries into 
#    comparative Excel/CSV reports for cross-model analysis.
#################################################################################


set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

STATIC_CONFIG="config/user_config/static_config.json"
CONFIG_FILE="config/user_config/dynamic_config.json"

# Helper function to print status messages
print_status_message() {
  local MESSAGE="$1"
  echo -e "${CYAN}>>> ${MESSAGE}${NC}"
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}vLLM Benchmark Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if config.json exists
if [ ! -f "${CONFIG_FILE}" ]; then
  echo -e "${RED}❌ Error: ${CONFIG_FILE} not found!${NC}"
  exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  echo -e "${RED}❌ Error: 'jq' is required but not installed.${NC}"
  exit 1
fi

# Read configuration from JSON
print_status_message "Reading configuration from ${CONFIG_FILE}"

FOLDER_PATH=$(jq -r '.filepath_of_the_project' "${STATIC_CONFIG}")
HF_TOKEN=$(jq -r '.huggingface_token' "${CONFIG_FILE}")
ENFORCE_EAGER=$(jq -r '.enforce_eager' "${CONFIG_FILE}")
VLLM_COMPILATION_MODE=$(jq -r '.VLLM_COMPILATION_MODE' "${CONFIG_FILE}")
RESULTS_PATH=$FOLDER_PATH/"Results"
ATTEN_BACKEND=$(jq -r '.VLLM_ATTENTION_BACKEND' "${CONFIG_FILE}")

# Check if folder already exists else skip folder creation
if [ ! -d "$RESULTS_PATH" ]; then
    MESSAGE="Results folder does not exist. Creating........."
    print_status_message "$MESSAGE"
    mkdir -p "$RESULTS_PATH"
else
    MESSAGE="Results folder already exists. Skipping creation"
    print_status_message "$MESSAGE"
fi


# # Docker configuration
# CUSTOM_IMAGE_NAME="vllm-benchmark-custom"
# CUSTOM_IMAGE_TAG="latest"
# FULL_CUSTOM_IMAGE="${CUSTOM_IMAGE_NAME}:${CUSTOM_IMAGE_TAG}"

# Custom scripts
#mnk frequency script
MNK_FREQ_SCRIPT="check_mnk.py"
#listing nvjets along with shape
NVJET_LIST="nvjet_list.py"
#sorting mnk based on frequency filename
MNK_SORT="mnk_sort.py"
#segregating open-closed source kernels
KERNEL_SEGREGATE="open_closed_kernel_listing.py"
#read all sorted mnk and aggregate into one
MNK_AGGREGATE="mnk_aggregate.py"
#read all open/closed source kernels and aggregate into one
KERNEL_AGGREGATE="kernel_aggregate.py"
#aggregates all nvjet details into one
NVJET_AGGREGATE="nvjet_aggregate.py"

# Filename to store mnk aggregation details
MNK_AGGREGATE_FILENAME=$RESULTS_PATH/"mnk_aggregate.csv"

# Filename to store open/close source kernel aggregation details
OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME=$RESULTS_PATH/"open_close_source_kernels_aggregate.xlsx"

# Filename to store nvjet aggregation details
NVJET_AGGREGATE_FILENAME=$RESULTS_PATH/"nvjet_aggregate.xlsx"

# Default details of available models
DEFAULT_MODEL_DETAILS_CSV=config/model_details/details.csv

# # GPU configuration
# DOCKER_GPU_FLAGS="${DOCKER_GPU_FLAGS:---gpus all}"

echo -e "${GREEN}✅ Configuration loaded${NC}"
echo ""

############### Display Configuration ###############
echo -e "${BLUE}Benchmark Configuration:${NC}"
# echo "  Models:                ${#MODEL_NAMES_ARRAY[@]} model(s)"
# echo "  Model List File:       ${MASTER_MODEL_LIST_PATH}"
echo "  Project Path:          ${FOLDER_PATH}"
echo "  Enforce Eager:         ${ENFORCE_EAGER}"
echo "  Compilation Mode:      ${VLLM_COMPILATION_MODE:-None}"
# echo "  Docker Image:          ${FULL_CUSTOM_IMAGE}"
echo ""

# # Check if custom image exists
# if ! docker images "${FULL_CUSTOM_IMAGE}" | grep -q "${CUSTOM_IMAGE_NAME}"; then
#   echo -e "${RED}❌ Error: Custom image '${FULL_CUSTOM_IMAGE}' not found!${NC}"
#   echo "Please run './build-custom-image.sh' first to build the image."
#   exit 1
# fi

############### Function to run benchmark for a single model ###############
run_model_benchmark() {
  local MODEL=$1
  local DATASET="${2:-}"   # optional argument (empty if not provided)
  local SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_' | tr ':' '_' | tr '.' '_')

  # Custom filenames 
  local CUBLAS_FILE_PATH="cublas_log.txt"
  local PROFILE_OUTPUT_NAME="profile_output"
  local MNK_OUTPUT="mnk_details.csv"
  local CUBLAS_FUNC_OUTPUT="cublas_func_calls.csv"
  local NSYS_KERNEL_SUMMARY="kernel_summary.csv"
  local KERNEL_RESULTS="nvjet_details.xlsx"
  local MNK_SORTED_OUTPUT="mnk_details_sorted.csv"
  local KERNELS_LIST="open_closed_source_kernels.xlsx"
  
  # Default results dir
  local RESULTS_DIR="${RESULTS_PATH}/${SAFE_MODEL_NAME}/Default_dataset"

  # If dataset provided, go one folder deeper
  if [[ -n "$DATASET" ]]; then
      RESULTS_DIR="${RESULTS_PATH}/${SAFE_MODEL_NAME}/Custom_dataset/${DATASET}"
  fi

  TARGET_DIR="vllm/benchmarks/results"

  if [ -d "$TARGET_DIR" ]; then
      echo "Found $TARGET_DIR. Deleting..."
      rm -rf "$TARGET_DIR"
  else
      echo "Directory $TARGET_DIR not found. Skipping."
  fi

  
  echo ""
  echo -e "${BLUE}==========================================${NC}"
  echo -e "${BLUE}Running benchmark for: ${MODEL}${NC}"
  echo -e "${BLUE}==========================================${NC}"
  echo "  Safe model name: ${SAFE_MODEL_NAME}"
  echo "  Output directory: $RESULTS_DIR"
  echo ""
  
  # docker run --rm \
  #   ${DOCKER_GPU_FLAGS} \
  #   -e HF_TOKEN="${HF_TOKEN}" \
  #   -e VLLM_ENFORCE_EAGER="${ENFORCE_EAGER}" \
  #   -e VLLM_COMPILATION_MODE="${VLLM_COMPILATION_MODE}" \
  #   --ipc=host \
  #   --shm-size=4g \
  #   -v "${FOLDER_PATH}":/tmp/workspace \
  #   -w /tmp/workspace \
  #   "${FULL_CUSTOM_IMAGE}" \
  #     bash -c "set -euo pipefail && \
  #     cd vllm && \
  #     pwd && \
  if [ -d ${RESULTS_DIR} ]; then rm -rf ${RESULTS_DIR}; fi
  mkdir -p ${RESULTS_DIR} 
  # LOG_DIR=\"${RESULTS_DIR}\"
  # LOGFILE=\"\$LOG_DIR/benchmark_\$(date +%Y%m%d_%H%M%S).log\"
  # exec >> "$LOGFILE" 2>&1
  cd vllm
  echo 'Starting benchmark for ${MODEL}...' 
  export HF_TOKEN=$HF_TOKEN
  export CUBLAS_LOGINFO_DBG=1 
  export CUBLAS_LOGDEST_DBG=${RESULTS_DIR}/${CUBLAS_FILE_PATH}
  # export attention-backend=$ATTEN_BACKEND
  nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --output ${RESULTS_DIR}/${PROFILE_OUTPUT_NAME} .buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh
  nsys stats --report cuda_gpu_kern_sum --format csv $RESULTS_DIR/$PROFILE_OUTPUT_NAME.nsys-rep > $RESULTS_DIR/$NSYS_KERNEL_SUMMARY 
  cd .. 
  python3 scripts/post_process_profiling/test_results_copy.py --input "$TARGET_DIR/benchmark_results.json" --output-dir "$RESULTS_DIR/perf_benchmark"
  python3 scripts/post_process_profiling/$MNK_FREQ_SCRIPT $RESULTS_DIR/$CUBLAS_FILE_PATH $RESULTS_DIR/$MNK_OUTPUT $RESULTS_DIR/$CUBLAS_FUNC_OUTPUT 
  python3 scripts/post_process_profiling/$NVJET_LIST $RESULTS_DIR/$NSYS_KERNEL_SUMMARY $RESULTS_DIR/$KERNEL_RESULTS 
  python3 scripts/post_process_profiling/$MNK_SORT $RESULTS_DIR/$MNK_OUTPUT $RESULTS_DIR/$MNK_SORTED_OUTPUT 
  python3 scripts/post_process_profiling/$KERNEL_SEGREGATE $RESULTS_DIR/$NSYS_KERNEL_SUMMARY $RESULTS_DIR/$KERNELS_LIST 
  rm -rf $RESULTS_DIR/$PROFILE_OUTPUT_NAME".nsys-rep"
  rm -rf $RESULTS_DIR/$PROFILE_OUTPUT_NAME".sqlite"
  rm -rf $RESULTS_DIR/$CUBLAS_FILE_PATH
  
  
  if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Benchmark complete for ${MODEL}${NC}"
  else
    echo ""
    echo -e "${RED}❌ Benchmark failed for ${MODEL}${NC}"
    return 1
  fi
}

############### Main Execution ###############
main() {
  # Default model details
  MODEL_CONFIG="models.yaml"

  # Check if yq is installed
  if ! command -v yq &> /dev/null; then
      echo "Error: yq is not installed."
      exit 1
  fi

  local CURRENT=0
  local FAILED_MODELS=()
  local START_TIME=$(date +%s)


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

  j=0
  CUST_CONFIG_PATH=$(yq -r ".path[$j].custom_config_path" "$MODEL_CONFIG")

  # Check if path is null or empty (yq -r returns 'null' for empty fields)
  if [[ "$CUST_CONFIG_PATH" != "null" && -n "$CUST_CONFIG_PATH" ]]; then
      echo "SUCCESS: Custom config path detected: $CUST_CONFIG_PATH"
      
      if [[ -f "$CUST_CONFIG_PATH" ]]; then
          echo "File exists. Proceeding with CSV parsing..."
          MODEL_DETAILS_CSV=$CUST_CONFIG_PATH

      else
          echo "ERROR: File path provided but file does not exist at: $CUST_CONFIG_PATH"
          exit 1
      fi
  else
      echo "Custom config file not found, using default config.........."
      MODEL_DETAILS_CSV=$DEFAULT_MODEL_DETAILS_CSV
  fi
  
  # Iterate through each model in the array
  for ((i=0; i<$NUM_MODELS; i++)); do
    MODEL=$(yq -r ".models[$i].name" "$MODEL_CONFIG")

    MODEL_NAME_NO_CASE_CHANGE=$MODEL
    CURRENT=$((CURRENT + 1))

    MODEL=$(echo $MODEL | tr -d '"')  # remove quotes

    # Convert model name to lowercase
    MODEL_NAME=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')


    echo "------------------------------------------------"
    echo "Processing Model: $MODEL_NAME"


    DATASET_MODE=$(yq -r '.datasets.mode // "default"' "$MODEL_CONFIG")

    if [[ "$DATASET_MODE" == "custom" ]]; then

        ############################
        # Load evaluation settings
        ############################

        NUM_PROMPTS=$(yq -r '.datasets.custom.evaluation.num_prompts // 200' "$MODEL_CONFIG")
        MAX_TOKENS=$(yq -r '.datasets.custom.evaluation.output_length // 128' "$MODEL_CONFIG")

        if [[ "$NUM_PROMPTS" -le 0 ]]; then
            echo "ERROR: evaluation.num_prompts must be > 0"
            exit 1        
        fi

        mapfile -t AVAILABLE_DATASETS < <(yq -r '.datasets.custom.available[]' "$MODEL_CONFIG")
        mapfile -t SELECTED_DATASETS < <(yq -r '.datasets.custom.selected // [] | .[]' "$MODEL_CONFIG")

        if [[ ${#SELECTED_DATASETS[@]} -eq 0 ]]; then
            echo "ERROR: Custom dataset mode enabled but no datasets selected."
            exit 1
        fi

        # Validate selected datasets
        for ds in "${SELECTED_DATASETS[@]}"; do
            if [[ ! " ${AVAILABLE_DATASETS[*]} " =~ " ${ds} " ]]; then
                echo "ERROR: Invalid dataset selected -> $ds"
                # exit 1
            fi
        done

        echo "Using custom datasets"
        if [[ ${#SELECTED_DATASETS[@]} -gt 0 ]]; then
            echo "Found ${#SELECTED_DATASETS[@]} datasets. Starting benchmark loop..."

            # 3. Loop over the selected list
            for DATASET in "${SELECTED_DATASETS[@]}"; do
                echo "------------------------------------------"
                echo 
                echo -e "${YELLOW}========================================${NC}"
                echo -e "${YELLOW}"Processing dataset: $DATASET"${NC}"
                echo -e "${YELLOW}========================================${NC}"
                python scripts/custom_test_config_modify/test_configuration_modify.py $MODEL
                python scripts/custom_test_config_modify/setup_vllm_benchmark.py \
                --from-benchmark-configs-dir config/test_configurations \
                --to-benchmark-configs-dir vllm/.buildkite/performance-benchmarks/tests \
                --models "${MODEL_NAME}" \
                --device "cuda"
                python scripts/custom_test_config_modify/add_json_values.py "$MODEL" "$VLLM_COMPILATION_MODE" "$ENFORCE_EAGER" "$FOLDER_PATH" "$MODEL_DETAILS_CSV" "$ATTEN_BACKEND" "$DATASET" "$NUM_PROMPTS" "$MAX_TOKENS"

                # Skip empty lines (extra safety)
                if [ -z "$MODEL" ]; then
                  continue
                fi
                
                echo -e "${YELLOW}========================================${NC}"
                echo -e "${YELLOW}Progress: ${CURRENT}/${NUM_MODELS}${NC}"
                echo -e "${YELLOW}========================================${NC}"
              
                if ! run_model_benchmark "${MODEL}" "$DATASET"; then
                  FAILED_MODELS+=("${MODEL}")
                fi
            done

        else
            echo "Notice: SELECTED_DATASETS is empty. No work to perform."
        fi


    else
        SELECTED_DATASETS=("default")
        echo "Using default dataset."
        python scripts/custom_test_config_modify/test_configuration_modify.py $MODEL_NAME_NO_CASE_CHANGE
        python scripts/custom_test_config_modify/setup_vllm_benchmark.py \
              --from-benchmark-configs-dir config/test_configurations \
              --to-benchmark-configs-dir vllm/.buildkite/performance-benchmarks/tests \
              --models "${MODEL_NAME}" \
              --device "cuda"
        python scripts/custom_test_config_modify/add_json_values.py "$MODEL" "$VLLM_COMPILATION_MODE" "$ENFORCE_EAGER" "$FOLDER_PATH" "$MODEL_DETAILS_CSV" "$ATTEN_BACKEND"

        # Skip empty lines (extra safety)
        if [ -z "$MODEL" ]; then
          continue
        fi
        
        echo -e "${YELLOW}========================================${NC}"
        echo -e "${YELLOW}Progress: ${CURRENT}/${NUM_MODELS}${NC}"
        echo -e "${YELLOW}========================================${NC}"
      
        if ! run_model_benchmark "${MODEL}"; then
          FAILED_MODELS+=("${MODEL}")
        fi

    fi
  done
  
  python scripts/post_process_profiling/$MNK_AGGREGATE $RESULTS_PATH $MNK_AGGREGATE_FILENAME
  python scripts/post_process_profiling/$KERNEL_AGGREGATE $RESULTS_PATH $OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME
  python scripts/post_process_profiling/$NVJET_AGGREGATE $RESULTS_PATH $NVJET_AGGREGATE_FILENAME

  local END_TIME=$(date +%s)
  local ELAPSED_TIME=$((END_TIME - START_TIME))
  local ELAPSED_MINUTES=$((ELAPSED_TIME / 60))
  local ELAPSED_SECONDS=$((ELAPSED_TIME % 60))
  
  echo ""
  echo -e "${GREEN}==========================================${NC}"
  echo -e "${GREEN}Benchmark Summary${NC}"
  echo -e "${GREEN}==========================================${NC}"
  echo "Total models:     ${NUM_MODELS}"
  echo "Successful:       $((NUM_MODELS - ${#FAILED_MODELS[@]}))"
  echo "Failed:           ${#FAILED_MODELS[@]}"
  echo "Total time:       ${ELAPSED_MINUTES}m ${ELAPSED_SECONDS}s"
  echo "Results location: ${FOLDER_PATH}"
  echo ""
  
  if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo -e "${RED}Failed models:${NC}"
    for model in "${FAILED_MODELS[@]}"; do
      echo "  - ${model}"
    done
    echo ""
    exit 1
  else
    echo -e "${GREEN}✅ All benchmarks completed successfully!${NC}"
  fi
}

main
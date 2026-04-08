#!/bin/bash

# ==============================================================================
# PURPOSE: Whisper ASR Profiling & Kernel Analysis Worker
# ------------------------------------------------------------------------------
# This script performs deep performance profiling for Transformer-based Whisper
# models. It is designed to be called by the master_launcher.sh but can run
# independently for a single model.
#
# Key Functions:
#   1. Orchestrates NVIDIA Nsys profiling for CUDA, cuDNN, and cuBLAS traces.
#   2. Captures cuBLAS logging data to extract GEMM shapes (M, N, K).
#   3. Automates GPU kernel analysis to identify open vs. closed source kernels.
#   4. Performs WER (Word Error Rate) scoring via a custom normalizer.
#   5. Cleans up heavy profiling artifacts (.nsys-rep, .sqlite) after analysis.
# ==============================================================================

# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================


# UPDATED: We now prioritize arguments passed from master_launcher.sh
# $1 is MODEL_ID, $2 is DATASET_FILE
MODEL_ID=${1:-"openai/whisper-large-v3"}
PASSED_DATASET_FILE=${2:-"datasets.txt"}

BATCH_SIZE=64
DEVICE_ID=0

# Paths and Filenames
#RESULTS_PATH=$(pwd)
RESULTS_PATH="$(dirname "$(pwd)")"
REPORT_FILENAME="model_nsys_report"
KERNEL_SUMMARY_FILENAME="kernel_summary.csv"
CUBLAS_LOG_NAME="cublas_log.txt"

# Post-Processing Python Scripts
SCRIPT_DIR="$RESULTS_PATH/post_processing_scripts"
KERNEL_LISTING_FILE="$SCRIPT_DIR/open_closed_kernel_listing.py"
MNK_FREQ_SCRIPT="$SCRIPT_DIR/check_mnk.py"
NVJET_LIST="$SCRIPT_DIR/nvjet_list.py"
MNK_SORT="$SCRIPT_DIR/mnk_sort.py"
MNK_AGGREGATE="$SCRIPT_DIR/mnk_aggregate.py"
KERNEL_AGGREGATE="$SCRIPT_DIR/kernel_aggregate.py"
NVJET_AGGREGATE="$SCRIPT_DIR/nvjet_aggregate.py"

# Environment Setup
export PYTHONPATH="..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# ==============================================================================
# 2. VALIDATION & LOADING
# ==============================================================================
# Check if the dataset file exists (either local or absolute path from master)
if [ ! -f "$PASSED_DATASET_FILE" ]; then
    echo " Error: Dataset file $PASSED_DATASET_FILE is missing."
    exit 1
fi

# Load datasets (ignoring comments and empty lines)
mapfile -t DATASETS_TO_RUN < <(grep -vE '^(\s*#|\s*$)' "$PASSED_DATASET_FILE")

echo " Benchmarking Model: $MODEL_ID"
echo " Loaded ${#DATASETS_TO_RUN[@]} datasets from $PASSED_DATASET_FILE"

# ==============================================================================
# 3. MAIN EXECUTION (Single Model Mode)
# ==============================================================================
# We no longer need the 'for MODEL_ID in MODEL_IDs' loop because 
# the master launcher handles the model iteration.


for ds in "${DATASETS_TO_RUN[@]}"; do
    # Determine data splits (LibriSpeech uses clean/other)
    if [ "$ds" == "librispeech" ] || [ "$ds" == "librispeech_asr" ]; then
        splits=("test.clean" "test.other")
    else
        splits=("test")
    fi

    # Clean name (just the model)
    CLEAN_NAME="${MODEL_ID##*/}"
    # Safe name for directories (openai/whisper -> openai--whisper)
    SAFE_MODEL_NAME="${MODEL_ID//\//--}"

    MODEL_PROF_RESULTS_PATH="$RESULTS_PATH/Results/$SAFE_MODEL_NAME/Profile_Details/$ds"
    mkdir -p "$MODEL_PROF_RESULTS_PATH"

    echo "===================================================="
    echo "STARTING PROFILING: $MODEL_ID"
    echo "DIR: $MODEL_PROF_RESULTS_PATH"
    echo "===================================================="


    # Check and delete existing cuBLAS log file
    if [ -f "$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME" ]; then
        rm -f "$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME"
        MESSAGE="Existing cuBLAS log file deleted"
        echo "$MESSAGE"
    else
        MESSAGE="No existing cuBLAS log file found. Proceeding"
        echo "$MESSAGE"
    fi

    # Set cuBLAS logging specifically for this model
    export CUBLAS_LOGINFO_DBG=1
    export CUBLAS_LOGDEST_DBG="$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME"

    # Check and delete existing .nsys-rep and .sqlite files
    if [ -d "$MODEL_PROF_RESULTS_PATH" ]; then

        # Find and delete .nsys-rep and .sqlite files if they exist
        find "$MODEL_PROF_RESULTS_PATH" -type f \( -name "*.nsys-rep" -o -name "*.sqlite" \) -exec rm -f {} +

        MESSAGE="Existing .nsys-rep and .sqlite files deleted (if present)"
        echo "$MESSAGE"

    else
        MESSAGE="Profiling results directory not found. Skipping cleanup"
        echo "$MESSAGE"
    fi


    for split in "${splits[@]}"; do
        START_SECONDS=$SECONDS
        echo "[$(date '+%H:%M:%S')] >>> Processing: $ds ($split)"
        
        # --- NSYS PROFILE ---
        nsys profile \
            --trace=cuda,cudnn,cublas,osrt \
            --output="$MODEL_PROF_RESULTS_PATH/${REPORT_FILENAME}" \
            --force-overwrite=true \
            --export=sqlite \
            python run_eval.py \
                --model_id="${MODEL_ID}" \
                --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
                --dataset="$ds" \
                --split="$split" \
                --device=${DEVICE_ID} \
                --batch_size=${BATCH_SIZE} \
                --max_eval_samples=-1

        DURATION=$(( SECONDS - START_SECONDS ))
        echo " DURATION: $((DURATION / 60))m $((DURATION % 60))s"
    done


    # ==============================================================================
    # 4. MODEL-LEVEL POST-PROCESSING
    # ==============================================================================
    echo " Generating reports for $CLEAN_NAME..."

    # Export GPU kernel summary to CSV (using the last generated report as representative)
    LAST_REPORT=$(ls -t "$MODEL_PROF_RESULTS_PATH"/*.nsys-rep | head -n 1)

    nsys stats --force-export true --report cuda_gpu_kern_sum --format csv \
        "$LAST_REPORT" > "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME"

    # Run Analysis Python Scripts
    python "$KERNEL_LISTING_FILE" "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME" "$MODEL_PROF_RESULTS_PATH/open_closed_source_kernels.xlsx"
    python "$MNK_FREQ_SCRIPT" "$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME" "$MODEL_PROF_RESULTS_PATH/mnk.csv" "$MODEL_PROF_RESULTS_PATH/cublas_func.csv"
    python "$NVJET_LIST" "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME" "$MODEL_PROF_RESULTS_PATH/nvjet_details.xlsx"
    python "$MNK_SORT" "$MODEL_PROF_RESULTS_PATH/mnk.csv" "$MODEL_PROF_RESULTS_PATH/mnk_sorted.csv"

    # Remove the files that are not required
    rm -rf $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.nsys-rep
    rm -rf $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.sqlite

    # Delete cuBLAS log file
    if [ -f "$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME" ]; then
        rm -f "$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME"
        MESSAGE="Cublas log file deleted"
        echo "$MESSAGE"
    else
        MESSAGE="No Cublas log file found"
        echo "$MESSAGE"
    fi

done

# Run WER Scoring (Normalizer)
echo " Calculating WER scores..."
# We use -P to handle the directory change safely
pushd ../normalizer > /dev/null
python -c "import eval_utils; eval_utils.score_results('${RESULTS_PATH}/results', '${MODEL_ID}')"
popd > /dev/null

echo " Benchmarking for $MODEL_ID finished."
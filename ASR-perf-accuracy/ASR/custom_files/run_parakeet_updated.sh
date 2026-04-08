#!/bin/bash

# ==============================================================================
# PURPOSE: Parakeet ASR Profiling & Kernel Analysis Worker
# ------------------------------------------------------------------------------
# This script performs deep performance profiling for Parakeet-based models.
#
# Key Functions:
#   1. Orchestrates NVIDIA Nsys profiling for CUDA, cuDNN, and cuBLAS traces.
#   2. Captures cuBLAS logging data to extract GEMM shapes (M, N, K).
#   3. Automates GPU kernel analysis to identify open vs. closed source kernels.
#   4. Performs WER (Word Error Rate) scoring via a custom normalizer.
#   5. Aggregates profiling metrics (MNK frequencies, NVJET details) into 
#      global cross-model reports.
# ==============================================================================

# --- CONFIGURATION & GLOBAL VARIABLES ---
DATASET_FILE="../datasets.txt"
# Add desired Parakeet models here
MODEL_IDs=("nvidia/parakeet-tdt-0.6b-v3") 

BATCH_SIZE=128
DEVICE_ID=0

# Define Paths & Filenames
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

export PYTHONPATH="..":$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# Validate dataset file
if [ ! -f "$DATASET_FILE" ]; then
    echo "Error: $DATASET_FILE not found. Please create it with one dataset per line (e.g., ami, voxpopuli)."
    exit 1
fi

mapfile -t DATASETS_TO_RUN < <(grep -vE '^(\s*#|\s*$)' "$DATASET_FILE")

# --- START PROCESSING ---
for MODEL_ID in "${MODEL_IDs[@]}"; do
    # Create a safe name for filesystem (replace / with --)
    SAFE_MODEL_NAME="${MODEL_ID//\//--}"
    
    for ds in "${DATASETS_TO_RUN[@]}"; do
        # Handle LibriSpeech splits automatically
        if [ "$ds" == "librispeech" ]; then
            splits=("test.clean" "test.other")
        else
            splits=("test")
        fi

        # Define and Create Profiling Directory
        MODEL_PROF_RESULTS_PATH="$RESULTS_PATH/Results/$SAFE_MODEL_NAME/Profile_Details/$ds"
        mkdir -p "$MODEL_PROF_RESULTS_PATH"

        echo "===================================================="
        echo "STARTING PROFILING: $MODEL_ID"
        echo "RESULTS DIR: $MODEL_PROF_RESULTS_PATH"
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

        # Set cuBLAS logging for this model
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
                    --model_id=${MODEL_ID} \
                    --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
                    --dataset="$ds" \
                    --split="$split" \
                    --device=${DEVICE_ID} \
                    --batch_size=${BATCH_SIZE} \
                    --max_eval_samples=-1

            DURATION=$(( SECONDS - START_SECONDS ))
            echo " DURATION: $((DURATION / 60))m $((DURATION % 60))s"
        done

         # --- POST-PROCESSING (Inside Model Loop) ---
        echo "Generating stats and kernel reports for $MODEL_ID..."
        
        # Generate CSV summary from Nsys report
        nsys stats --force-export true --report cuda_gpu_kern_sum --format csv \
            "$MODEL_PROF_RESULTS_PATH/${REPORT_FILENAME}.nsys-rep" > "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME"

        # Run Analysis Scripts
        python $KERNEL_LISTING_FILE "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME" "$MODEL_PROF_RESULTS_PATH/open_closed_source_kernels.xlsx"
        python $MNK_FREQ_SCRIPT "$MODEL_PROF_RESULTS_PATH/$CUBLAS_LOG_NAME" "$MODEL_PROF_RESULTS_PATH/mnk.csv" "$MODEL_PROF_RESULTS_PATH/cublas_func.csv"
        python $NVJET_LIST "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME" "$MODEL_PROF_RESULTS_PATH/nvjet_details.xlsx"
        python $MNK_SORT "$MODEL_PROF_RESULTS_PATH/mnk.csv" "$MODEL_PROF_RESULTS_PATH/mnk_sorted.csv"

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

        # Final Scoring (Normalizer)
        echo "Calculating final WER scores..."
        cd ../normalizer
        python -c "import eval_utils; eval_utils.score_results('${RESULTS_PATH}/results', '${MODEL_ID}')"

    done

done

cd "$RESULTS_PATH"
# --- GLOBAL AGGREGATION (Outside Model Loop) ---
echo "===================================================="
echo "AGGREGATING ALL MODEL RESULTS"
echo "===================================================="
MNK_AGG_FILE="$RESULTS_PATH/Results/mnk_aggregate.csv"
KERNEL_AGG_FILE="$RESULTS_PATH/Results/open_close_source_kernels_aggregate.xlsx"
NVJET_AGG_FILE="$RESULTS_PATH/Results/nvjet_aggregate.xlsx"

python $MNK_AGGREGATE "$RESULTS_PATH/Results" "$MNK_AGG_FILE"
python $KERNEL_AGGREGATE "$RESULTS_PATH/Results" "$KERNEL_AGG_FILE"
python $NVJET_AGGREGATE "$RESULTS_PATH/Results" "$NVJET_AGG_FILE"

echo "All tasks completed successfully."
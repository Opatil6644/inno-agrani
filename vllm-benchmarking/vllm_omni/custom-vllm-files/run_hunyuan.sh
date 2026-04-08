#!/bin/bash

#################################################################################
# PURPOSE: Automated Model Inference and NVIDIA Profiling Pipeline (Centralized)
#################################################################################

# 1. PATH SETUP
# -------------------------------------------------------------------------------
# Identifying the Base Directory (3 levels up from examples/offline_inference/...)
BASE_DIR=$(realpath ../../../)
MODEL_NAME=$(basename "$PWD")

# Global Results path (where aggregation happens)
RESULTS_PATH="$BASE_DIR"

# Specific subfolder for this model to avoid overwriting other models
MODEL_PROF_RESULTS_PATH="$RESULTS_PATH/Results/$MODEL_NAME"

# Create the directories
mkdir -p "$MODEL_PROF_RESULTS_PATH"

echo "Profiling results will be saved to: $MODEL_PROF_RESULTS_PATH"

# 2. SCRIPT PATHS (Remaining as relative jump to base)
# -------------------------------------------------------------------------------
KERNEL_LISTING_FILE="../../../post_processing_scripts/open_closed_kernel_listing.py"
MNK_AGGREGATE="../../../post_processing_scripts/mnk_aggregate.py"
KERNEL_AGGREGATE="../../../post_processing_scripts/kernel_aggregate.py"
NVJET_AGGREGATE="../../../post_processing_scripts/nvjet_aggregate.py"
MNK_FREQ_SCRIPT="../../../post_processing_scripts/check_mnk.py"
NVJET_LIST="../../../post_processing_scripts/nvjet_list.py"
MNK_SORT="../../../post_processing_scripts/mnk_sort.py"

# 3. FILENAME DEFINITIONS
# -------------------------------------------------------------------------------
# Aggregation files (located in the main Results folder)
MNK_AGGREGATE_FILENAME="$RESULTS_PATH/Results/mnk_aggregate.csv"
OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME="$RESULTS_PATH/Results/open_close_source_kernels_aggregate.xlsx"
NVJET_AGGREGATE_FILENAME="$RESULTS_PATH/Results/nvjet_aggregate.xlsx"

# Model-specific filenames
REPORT_FILENAME="model_nsys_report"
KERNEL_SUMMARY_FILENAME="kernel_summary.csv"
CUBLAS_FILE_PATH="$MODEL_PROF_RESULTS_PATH/cublas_log.txt"
CUBLAS_FUNC_OUTPUT="$MODEL_PROF_RESULTS_PATH/cublas_func.csv"
MNK_OUTPUT="$MODEL_PROF_RESULTS_PATH/mnk.csv"
MNK_SORTED_OUTPUT="$MODEL_PROF_RESULTS_PATH/mnk_sorted.csv"
KERNEL_RESULTS="$MODEL_PROF_RESULTS_PATH/nvjet_details.xlsx"

# 4. ENVIRONMENT & PROFILING
# -------------------------------------------------------------------------------
export CUBLAS_LOGINFO_DBG=1
export CUBLAS_LOGDEST_DBG="$CUBLAS_FILE_PATH"

nsys profile \
  --trace cuda,nvtx,osrt,cudnn,cublas \
  --output "$MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME" \
  --force-overwrite true \
  --export=sqlite \
python image_to_text.py \
  --image cherry_blossom.jpg \
  --prompt "<|startoftext|>You are an assistant that understands images and outputs text.<img>Describe the content of the picture."

# 5. POST-PROCESSING
# -------------------------------------------------------------------------------
# Convert stats to CSV
nsys stats --force-export true --report cuda_gpu_kern_sum --format csv \
    "$MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME.nsys-rep" > "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME"

# Run Python analysis scripts
python "$KERNEL_LISTING_FILE" "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME" "$MODEL_PROF_RESULTS_PATH/open_closed_source_kernels.xlsx"
python "$MNK_FREQ_SCRIPT" "$CUBLAS_FILE_PATH" "$MNK_OUTPUT" "$CUBLAS_FUNC_OUTPUT"
python "$NVJET_LIST" "$MODEL_PROF_RESULTS_PATH/$KERNEL_SUMMARY_FILENAME" "$KERNEL_RESULTS"
python "$MNK_SORT" "$MNK_OUTPUT" "$MNK_SORTED_OUTPUT"

# 6. AGGREGATION
# -------------------------------------------------------------------------------
# Note: The aggregation scripts take $RESULTS_PATH which is now the base dir.
python "$MNK_AGGREGATE" "$RESULTS_PATH" "$MNK_AGGREGATE_FILENAME"
python "$KERNEL_AGGREGATE" "$RESULTS_PATH" "$OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME"
python "$NVJET_AGGREGATE" "$RESULTS_PATH" "$NVJET_AGGREGATE_FILENAME"
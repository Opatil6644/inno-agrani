#!/bin/bash
##############################################
# EvalGIM Model Evaluation Pipeline
#
# This script automates the end-to-end evaluation workflow with nsys profiling with latency and throughput tests:
#
# 1. Loads model names from master model list and configurations from config.json.
# 2. Ensures required result directories exist.
# 3. For each model:
#       - Cleans previously generated data
#       - Generates images
#       - Copies generated images to a Results/Generated folder
#       - runs nsys profiling
#       - runs latency and throughput tests
#       - Runs evaluation
#       - Copies evaluation metrics to Results/Evaluation
#       - Cleans temporary data folders
##############################################

# Exit immediately if a command exits with a non-zero status
set -e  

############### Helper: Check config.json exists ###############
# Absolute path of the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_JSON_PATH="$SCRIPT_DIR/../../config/config.json"

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
    echo "Launching evaluation + profiling for MODEL: $text"
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

# System path where EvalGIM cloned repo exists
SYS_PATH=$(jq -r '.sys_path' "$CONFIG_JSON_PATH")

# Path where evaluation results will be stored
RESULTS_PATH=$(jq -r '.results_path' "$CONFIG_JSON_PATH")

# Number of samples to generate per model
NUM_SAMPLES=$(jq -r '.num_samples' "$CONFIG_JSON_PATH")

# Batch-Size used to generate images
BATCH_SIZE=$(jq -r '.batch_size' "$CONFIG_JSON_PATH")

# Whether to save generated images
SAVE_GEN_IMG=$(jq -r '.save_gen_img' "$CONFIG_JSON_PATH")

# Open/close source custom python filename
KERNEL_LISTING_FILE="open_closed_kernel_listing.py"

# Combining latency and throughput results python filename
COMBINE_LATENCY_THROUGHPUT="combine_latency_throughput.py"


#read all sorted mnk and aggregate into one
MNK_AGGREGATE="mnk_aggregate.py"

#read all open/closed source kernels and aggregate into one
KERNEL_AGGREGATE="kernel_aggregate.py"

#aggregates all nvjet details into one
NVJET_AGGREGATE="nvjet_aggregate.py"

# Latency and throughput consolidated filename
CONSOLIDATED_FILENAME="latency_throughput_summary.csv"

# Filename to store mnk aggregation details
MNK_AGGREGATE_FILENAME=$RESULTS_PATH/"mnk_aggregate.csv"

# Filename to store open/close source kernel aggregation details
OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME=$RESULTS_PATH/"open_close_source_kernels_aggregate.xlsx"

# Filename to store nvjet aggregation details
NVJET_AGGREGATE_FILENAME=$RESULTS_PATH/"nvjet_aggregate.xlsx"

############### Loop over model names ###############
# Counter to keep track of models
COUNTER=0

for MODEL_NAME in "${MODEL_NAMES_ARRAY[@]}"; do
  # MODEL_NAME usually contains quotes → strip them
  MODEL_NAME=$(echo $MODEL_NAME | tr -d '"')

  print_box "$MODEL_NAME"

  # Replace / or \ with -- so it becomes a safe folder name
  PRE_PROCESSED_MODEL_NAME=${MODEL_NAME//[\/\\]/--}

  # Model specific folder to save results
  MODEL_RESULTS_DIR_PATH="$RESULTS_PATH/Results/$PRE_PROCESSED_MODEL_NAME"

  # Mnk frequency script
  MNK_FREQ_SCRIPT="check_mnk.py"

  # Listing nvjets along with shape
  NVJET_LIST="nvjet_list.py"

  # Sorting mnk based on frequency filename
  MNK_SORT="mnk_sort.py"

  ############### Prepare output directories ###############

  # Check whether to save generated images
  if [ "$SAVE_GEN_IMG" = "true" ]; then
    # Folder to store generated images
    DIR_PATH="$MODEL_RESULTS_DIR_PATH/Generated_Images"

    # Check if folder already exists else skip folder creation
    if [ ! -d "$DIR_PATH" ]; then
        MESSAGE="Results folder does not exist. Creating for storing generated images"
        print_status_message "$MESSAGE"
        mkdir -p "$DIR_PATH"
    else
        MESSAGE="Generated folder already exists. Skipping creation"
        print_status_message "$MESSAGE"
    fi

  else
    MESSAGE="Not storing images as it disabled. Skipping folder creation"
    print_status_message "$MESSAGE"
  fi

  # Folder to store evaluation results
  DIR_PATH_EVALUATION="$MODEL_RESULTS_DIR_PATH/Evaluation"

  # Check if folder already exists else skip folder creation
  if [ ! -d "$DIR_PATH_EVALUATION" ]; then
      MESSAGE="Results folder does not exist. Creating for storing evaluation results"
      print_status_message "$MESSAGE"
      mkdir -p "$DIR_PATH_EVALUATION"
  else
      MESSAGE="Evaluation folder already exists. Skipping creation"
      print_status_message "$MESSAGE"
  fi


  # Folder path to profiling reports
  MODEL_PROF_RESULTS_PATH="$MODEL_RESULTS_DIR_PATH/Profile_Details"

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

  # Delete cuBLAS log file
  if [ -f "$CUBLAS_FILE_PATH" ]; then
      rm -f "$CUBLAS_FILE_PATH"
      MESSAGE="Cublas log file deleted"
      print_status_message "$MESSAGE"
  else
      MESSAGE="No Cublas log file found"
      print_status_message "$MESSAGE"
  fi


  # Environment variables set to capture cublas logs
  export CUBLAS_LOGINFO_DBG=1
  export CUBLAS_LOGDEST_DBG=$CUBLAS_FILE_PATH


  # Check and delete existing .nsys-rep and .sqlite files
  if [ -d "$MODEL_PROF_RESULTS_PATH" ]; then

      # Find matching files and store them
      FOUND_FILES=$(find "$MODEL_PROF_RESULTS_PATH" -type f \( -name "*.nsys-rep" -o -name "*.sqlite" \))

      if [ -n "$FOUND_FILES" ]; then
          # Files found → delete them
          rm -f $FOUND_FILES

          MESSAGE="Found and deleted the following profiling files:"
          print_status_message "$MESSAGE"
          echo "$FOUND_FILES"
      else
          MESSAGE="No .nsys-rep or .sqlite files found. Nothing to delete"
          print_status_message "$MESSAGE"
      fi

  else
      MESSAGE="Profiling results directory not found. Skipping cleanup"
      print_status_message "$MESSAGE"
  fi


  #####################################################
  # Part 1: Clean old generated data for the model
  #####################################################

  # By default EvalGIM stores generated images inside EvalGIM->projects->generated->model_name->
  TARGET_FOLDER="$SYS_PATH/projects/generated/${PRE_PROCESSED_MODEL_NAME}__coco_txt_dataset__cfg7.5"

  # Check and delete folder if already exists
  if [ -d "$TARGET_FOLDER" ]; then
      rm -rf "$TARGET_FOLDER"
      MESSAGE="Older generated data deleted successfully"
    print_status_message "$MESSAGE"

  else
      MESSAGE="Older generated data not found"
    print_status_message "$MESSAGE"
  fi
  
  #####################################################
  # Part 2: Enter EvalGIM directory only once
  #####################################################
  
  # Change directory only for the first model inside the list
  if [[ $COUNTER -eq 0 ]]; then
    MESSAGE="Entering EvalGIM directory (first model only)..."
    print_status_message "$MESSAGE"
    cd EvalGIM || { echo "ERROR: Failed to cd EvalGIM"; exit 1; }
        
  else
    MESSAGE="Skipping directory change because already inside EvalGIM directory"
    print_status_message "$MESSAGE"
  fi

  # increment counter
  COUNTER=$((COUNTER + 1))

  #####################################################
  # Part 3: Generate Images
  #####################################################

  MESSAGE="Generating Images....................."
  print_status_message "$MESSAGE"

  # Nsys profile command used to generate nsys profiling report
  nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --force-overwrite true --output $MODEL_PROF_RESULTS_PATH/$REPORT_FILENAME  python -m evaluation_library.generate \
  --model_id $MODEL_NAME \
  --num_samples $NUM_SAMPLES \
  --batch_size $BATCH_SIZE \
  --local

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


  MESSAGE="Image generation completed"
  print_status_message "$MESSAGE"

  #####################################################
  # Part 4: Copy Generated Images to Results Folder
  #####################################################
 
  # Check whether to copy generated images
  if [ "$SAVE_GEN_IMG" = "true" ]; then
    SOURCE_FOLDER=$TARGET_FOLDER

    # Folder path to store/copy generated images
    DEST_FOLDER="${DIR_PATH}"

    # Remove existing folder
    rm -rf "$DEST_FOLDER"        

    # Create destination folder if not exists
    mkdir -p "$DEST_FOLDER"

    # Copy source folder contents
    cp -r "$SOURCE_FOLDER"/. "$DEST_FOLDER"/


    MESSAGE="Copied generated images to results folder"
    print_status_message "$MESSAGE"

  else
    MESSAGE="Skipped copying generated images as it is disabled"
    print_status_message "$MESSAGE"
  fi

  # Copy latency and throughput results 
  cp $SYS_PATH/projects/generated/latency.json $MODEL_RESULTS_DIR_PATH
  cp $SYS_PATH/projects/generated/throughput.json $MODEL_RESULTS_DIR_PATH

  MESSAGE="Copied latency and throughput metrics to results folder"
  print_status_message "$MESSAGE"

   #####################################################
   # Part 5: Run Evaluation
   #####################################################

  MESSAGE="Model Evaluation started....................."
  print_status_message "$MESSAGE"

  python -m evaluation_library.evaluate \
  --model_id $MODEL_NAME \
  --local \
  --generated_images_path $TARGET_FOLDER \
  --marginal_metrics fid_torchmetrics,prdc \
  --conditional_metrics clipscore \

  MESSAGE="Model Evaluation completed"
  print_status_message "$MESSAGE"

  #####################################################
  # Part 6: Copy evaluation outputs
  #####################################################
  
  # By default EvalGIM stores evaluation metric results inside EvalGIM->projects->generated->evals->
  EVAL_PARENT_FOLDER=$SYS_PATH/projects/generated/evals

  # Folder path to store/copy evaluation results
  DESTINATION="${DIR_PATH_EVALUATION}"

  # Remove existing folder
  rm -rf "$DESTINATION"        

  # Pick the latest evaluation log folder
  LATEST_FOLDER=$(find "$EVAL_PARENT_FOLDER" -mindepth 1 -maxdepth 1 -type d | head -1)
  
  # Create destination directory if not exists
  mkdir -p "$DESTINATION"

  # Copy the folder
  cp -r "$LATEST_FOLDER/scores.yaml" "$DESTINATION"

  MESSAGE="Copied evaluation metrics to results folder"
  print_status_message "$MESSAGE"

  #####################################################
  # Part 7: Copy results.csv (common summary)
  #####################################################

  # ---- Copy CSV(aggregated results of all the runs) file (always one) ----
  # By default EvalGIM stores evaluation metric results inside EvalGIM->projects->generated->evals->
  CSV_FILE="$EVAL_PARENT_FOLDER/results.csv"

  # Folder path to store/copy evaluation(aggregated of all runs) results
  DESTINATION_CSV_FILE="$RESULTS_PATH/Results"

  # Check file and copy
  if [[ -f "$CSV_FILE" ]]; then
    cp "$CSV_FILE" "$DESTINATION_CSV_FILE"
    MESSAGE="CSV file copied to results folder"
    print_status_message "$MESSAGE"
  else
    MESSAGE="CSV file not found"
    print_status_message "$MESSAGE"
  fi

  #####################################################
  # Part 8: Cleanup temporary generation and evaluation folders
  #####################################################

  # Check and delete the generated images to avoid causing errors(because EvalGIM doesn't allow evaluating same model before cleaning its older run logs/outputs)
  if [ -d "$TARGET_FOLDER" ]; then
      rm -rf "$TARGET_FOLDER"
      MESSAGE="Cleaning image generated folder, deleted successfully"
      print_status_message "$MESSAGE"
  else
      MESSAGE="Generated images folder not found"
      print_status_message "$MESSAGE"

  fi

  # Check and delete the evaluation metric results generated to avoid causing errors(because EvalGIM doesn't allow evaluating same model before cleaning its older run logs/outputs)
  # Keeps results.csv because it has aggregated results of all model runs
  if [ -d "$EVAL_PARENT_FOLDER" ]; then
    MESSAGE="Cleaning evaluation metrics folder, (keeping results.csv). Deleted successfully"
    print_status_message "$MESSAGE"

    find "$EVAL_PARENT_FOLDER" -mindepth 1 ! -name "results.csv" -exec rm -rf {} +

    MESSAGE="Cleanup done."
    print_status_message "$MESSAGE"
  else
      MESSAGE="Evaluation metrics folder not found"
      print_status_message "$MESSAGE"
  fi

done


# Collate latency and throughput results for each model run into a csv file
python $COMBINE_LATENCY_THROUGHPUT --results_dir $RESULTS_PATH/Results --output_csv $RESULTS_PATH/Results/$CONSOLIDATED_FILENAME

python3 $MNK_AGGREGATE $RESULTS_PATH $MNK_AGGREGATE_FILENAME
python3 $KERNEL_AGGREGATE $RESULTS_PATH $OPEN_CLOSE_SOURCE_AGGREGATE_FILENAME
python3 $NVJET_AGGREGATE $RESULTS_PATH $NVJET_AGGREGATE_FILENAME
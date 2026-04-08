#!/bin/bash
##############################################
# Repo Checkout Script + COCO Dataset Setup 
# 
# This script automates the following:
# 1. Clones a GitHub repository and checks out a specific commit.
# 2. Updates configuration files using the repository path.
# 3. Installs the required packages
# 4. Downloads, extracts, and prepares the COCO 2014 validation dataset.
# 5. Updates paths in paths.yaml for COCO annotations and images.
# 6. Performs Hugging Face login.
# 7. Replaces CLIP model references inside customCLIPScore.py.
##############################################

# Exit immediately if a command exits with a non-zero status
set -e  

# --- Configuration ---
REPO_URL=https://github.com/facebookresearch/EvalGIM.git
COMMIT_SHA=d1c86f2089381d7e470ce43308174a6eff08f198

# Extract repo directory name (EvalGIM)
REPO_DIR=$(basename "$REPO_URL" .git)


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

##############################################
# 1. Clone the repository
##############################################

MESSAGE="Starting Clone Process"
print_status_message "$MESSAGE"
echo "Destination Directory: $REPO_DIR"

# We don't specify a branch because we just need the structure and then fetch the specific commit.
echo "Cloning repository"
git clone "$REPO_URL" --depth 1

# Abort if clone fails
if [ $? -ne 0 ]; then
    MESSAGE="Error: Initial clone failed. Exiting."
    print_status_message "$MESSAGE"
    exit 1
fi

##############################################
# 2. Enter repository directory
##############################################
echo ""
echo "Entering directory: $REPO_DIR"
cd "$REPO_DIR"

##############################################
# 3. Fetch + checkout specific commit
##############################################
echo ""
echo "Fetching specific commit: $COMMIT_SHA"
git fetch origin "$COMMIT_SHA"

# 5. Checkout the specific commit
echo ""
echo "Checking out commit: $COMMIT_SHA"
git checkout "$COMMIT_SHA"

MESSAGE="Clone Successful!"
print_status_message "$MESSAGE"

# Store absolute repo path
REPO_ABS_PATH=$(pwd)

# Return to parent folder
cd ..

# Store another copy absolute repo path
REPO_ABS_PATH_NEW=$(pwd)

# Replace custom .toml file with .toml file present in repo to prevent version issues
cp $REPO_ABS_PATH_NEW/pyproject.toml $REPO_ABS_PATH/pyproject.toml
echo "Requirements file copied"

# Replace custom_generate.py with standard EvalGIM generate.py 
# Following changes made in custom_generate.py:
#(*) Added latency and throughput calculations and saving back those results into json files
cp $REPO_ABS_PATH_NEW/scripts/custom_generate/generate.py $REPO_ABS_PATH/evaluation_library/generate.py
echo "Custom img generate python file copied"

# Copy custom python file used for listing out open/source kernels to EvalGIM dir
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/open_closed_kernel_listing.py $REPO_ABS_PATH
echo "Custom listing open/closed source kernels file copied"

# Copy consolidated python file to EvalGIM dir
cp $REPO_ABS_PATH_NEW/scripts/consolidate_perf_results/combine_latency_throughput.py $REPO_ABS_PATH
echo "Custom latency/throughput consolidated python file copied"


# Copy mnk frequency python script
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/check_mnk.py $REPO_ABS_PATH
echo "Custom mnk frequency python file copied"

# Copy nvjet listing python script along with shape
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/nvjet_list.py $REPO_ABS_PATH
echo "Custom nvjet listing python file copied"

# Copy sorting mnk frequency script
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/mnk_sort.py $REPO_ABS_PATH
echo "Custom mnk sorting python file copied"

# Copy mnk aggregate python script
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/mnk_aggregate.py $REPO_ABS_PATH
echo "Custom mnk aggregation python file copied"

# Copy open/close source kernels aggregate python script
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/kernel_aggregate.py $REPO_ABS_PATH
echo "Custom open/close source aggregation python file copied"

# Copy nvjet details aggregate python script
cp $REPO_ABS_PATH_NEW/scripts/post_process_profiling/nvjet_aggregate.py $REPO_ABS_PATH
echo "Custom nvjet details aggregation python file copied"

MESSAGE="Installing required packages.............."
print_status_message "$MESSAGE"

# Install the required packages
pip install -e $REPO_ABS_PATH
echo "Installation completed"

MESSAGE="Updating config file.............."
print_status_message "$MESSAGE"


##############################################
# 4. Update config.json with repo path
##############################################
CONFIG_FILE="$REPO_ABS_PATH_NEW/config/config.json"

# Call Python script to update sys.path inside config.json
python $REPO_ABS_PATH_NEW/scripts/setup/update_sys_path_config.py $REPO_ABS_PATH $CONFIG_FILE


##############################################
# 5. Read COCO dataset path from config.json
##############################################
COCO_DATA_PATH=$(jq -r '.coco_data_path' "$CONFIG_FILE")

if [[ -z "$COCO_DATA_PATH" ]]; then
    MESSAGE="ERROR: coco_data_path missing in config.json"
    print_status_message "$MESSAGE"
    exit 1
fi


# Paths to required files and folders
VAL_ZIP="$COCO_DATA_PATH/val2014.zip"
ANNOT_ZIP="$COCO_DATA_PATH/annotations_trainval2014.zip"
VAL_FOLDER="$COCO_DATA_PATH/val2014"
ANNOT_FOLDER="$COCO_DATA_PATH/annotations"

# Ensure directory exists
mkdir -p "$COCO_DATA_PATH"

##############################################
# 6. Download and extract COCO 2014 dataset
##############################################

# --- val2014 images ---
if [[ -d "$VAL_FOLDER" ]]; then
    MESSAGE="COCO val2014 exists skipping"
    print_status_message "$MESSAGE"
else
    MESSAGE="Downloading COCO val2014.zip..."
    print_status_message "$MESSAGE"
    wget -O "$VAL_ZIP" http://images.cocodataset.org/zips/val2014.zip
    unzip "$VAL_ZIP" -d "$COCO_DATA_PATH"
    rm "$VAL_ZIP"
    echo "COCO val data downloaded"
fi

# --- annotations ---
if [[ -d "$ANNOT_FOLDER" ]]; then
    MESSAGE="COCO Annotations exist skipping"
    print_status_message "$MESSAGE"
else
    MESSAGE="Downloading COCO annotations..."
    print_status_message "$MESSAGE"
    wget -O "$ANNOT_ZIP" http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip "$ANNOT_ZIP" -d "$COCO_DATA_PATH"
    rm "$ANNOT_ZIP"
    echo "COCO annotation data downloaded"
fi

MESSAGE="COCO dataset is ready!"
print_status_message "$MESSAGE"


##############################################
# 7. Update paths.yaml inside EvalGIM
##############################################

MESSAGE="Updating paths yaml file................."
print_status_message "$MESSAGE"

# yaml file path
YAML_FILE="$REPO_ABS_PATH/evaluation_library/data/paths.yaml"

COCO_ANNOT_VAL="$COCO_DATA_PATH/annotations/captions_val2014.json"
COCO_IMG_ROOT_VAL="$COCO_DATA_PATH/val2014"

# Update ANNOT_VAL
sed -i '/^COCO:/,/^GeoDE:/ {
    s|^\( *ANNOT_VAL: \).*$|\1"'"$COCO_ANNOT_VAL"'"|
}' "$YAML_FILE"

# Update IMG_ROOT_VAL
sed -i '/^COCO:/,/^GeoDE:/ {
    s|^\( *IMG_ROOT_VAL: \).*$|\1"'"$COCO_IMG_ROOT_VAL"'"|
}' "$YAML_FILE"

echo "Paths.yaml updated with COCO annotation + image paths"

##############################################
# 8. Hugging Face login
##############################################

MESSAGE="Logging into Hugging Face............."
print_status_message "$MESSAGE"

# Call Python script for hf access
python $REPO_ABS_PATH_NEW/scripts/setup/hf_login.py --config_json_path $CONFIG_FILE

##############################################
# 9. Replace CLIP model reference
##############################################


MESSAGE="Updating CLIP model............."
print_status_message "$MESSAGE"
# clipscore file path
CLIP_FILE_PATH="$REPO_ABS_PATH/evaluation_library/metrics/customCLIPScore.py"

# Replace the LAION CLIP model with openai/clip-vit-base-patch16 
# because the LAION model causes errors during EvalGIM evaluations
sed -i "s|laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K|openai/clip-vit-base-patch16|g" "$CLIP_FILE_PATH"
echo "CLIP model updation successful!"

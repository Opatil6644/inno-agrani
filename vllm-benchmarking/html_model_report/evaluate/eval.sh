# #!/bin/bash

# ################################################################################
# #Model Architecture Reporter
# # ------------------------------------------
# # This script automates the generation of model architecture reports by:
# # 1. Verifying dependencies (yq).
# # 2. Creating a centralized 'Results' directory for HTML output.
# # 3. Parsing 'models.yaml' to identify a list of target Hugging Face models.
# # 4. Sequentially executing the Python profiler for each model.
# ################################################################################

# YELLOW='\033[1;33m'
# NC='\033[0m'

# # Default model details
# MODEL_CONFIG="models.yaml"

# # Check if yq is installed
# if ! command -v yq &> /dev/null; then
#     echo "Error: yq is not installed."
#     exit 1
# fi

# RESULTS_DIR="Results"

# # Create results folder if it doesn't exist
# if [ ! -d "$RESULTS_DIR" ]; then
#     echo -e "${YELLOW}Creating directory: $RESULTS_DIR${NC}"
#     mkdir -p "$RESULTS_DIR"
# else
#     echo "Directory $RESULTS_DIR already exists. Skipping creation."
# fi


# # If the file is empty or 'models' key doesn't exist, yq returns 0 or null
# NUM_MODELS=$(yq -r '.models | length // 0' "$MODEL_CONFIG")

# # --- NEW CONDITION: Check if at least one model is selected ---
# if [[ "$NUM_MODELS" -eq 0 || "$NUM_MODELS" == "null" ]]; then
#     echo "------------------------------------------------"
#     echo "ERROR: No enabled models found in $MODEL_CONFIG."
#     echo "Please uncomment at least one model to proceed."
#     echo "------------------------------------------------"
#     exit 1
# fi


# echo -e "${YELLOW}Starting report preparation for ${NUM_MODELS} model(s)...${NC}"
# echo ""

# # Iterate through each model in the array
# for ((i=0; i<$NUM_MODELS; i++)); do
#   MODEL=$(yq -r ".models[$i].name" "$MODEL_CONFIG")

#   echo $MODEL

#   python evaluate/html_model_tree.py $MODEL $RESULTS_DIR

# done


#!/bin/bash

YELLOW='\033[1;33m'
NC='\033[0m'

MODEL_CONFIG="models.yaml"
RESULTS_DIR="Results"

# Check dependencies
if ! command -v yq &> /dev/null; then
    echo "Error: yq is not installed."
    exit 1
fi

# Folder creation logic
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${YELLOW}Creating directory: $RESULTS_DIR${NC}"
    mkdir -p "$RESULTS_DIR"
fi

# Count models
NUM_MODELS=$(yq -r '.models | length // 0' "$MODEL_CONFIG")

if [[ "$NUM_MODELS" -eq 0 || "$NUM_MODELS" == "null" ]]; then
    echo "------------------------------------------------"
    echo "ERROR: No enabled models found in $MODEL_CONFIG."
    echo "Please uncomment at least one model to proceed."
    echo "------------------------------------------------"
    exit 1
fi

echo -e "${YELLOW}Starting report preparation for ${NUM_MODELS} model(s)...${NC}"

# Iterate through models
for ((i=0; i<$NUM_MODELS; i++)); do
    MODEL=$(yq -r ".models[$i].name" "$MODEL_CONFIG")
    
    # Check if MODEL is empty or null before running python
    if [[ -z "$MODEL" || "$MODEL" == "null" ]]; then
        continue
    fi

    echo -e "${YELLOW}Processing: $MODEL${NC}"

    # Use quotes around variables to handle special characters
    python evaluate/html_model_tree.py "$MODEL" "$RESULTS_DIR"
done

# Typo was likely here: "cho" vs "echo"
echo -e "${YELLOW}Done${NC}"
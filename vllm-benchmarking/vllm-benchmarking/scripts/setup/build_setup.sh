#!/bin/bash

#################################################################################
# PURPOSE:
# This script automates the creation of a setup for vLLM 
# benchmarking. It ensures the build environment is synchronized with a specific 
# vLLM GitHub commit and injects custom performance-tracking scripts and datasets.
#
# KEY FUNCTIONALITIES:
# 1. Dependency Management: Automatically detects and installs 'jq' for JSON parsing.
# 2. Config-Driven Build: Reads the target 'commit_id' and 'project_path' from 
#    static_config.json to ensure reproducible builds.
# 3. Repository Synchronization: Clones or updates the official vLLM repository 
#    to the exact commit ID specified in the configuration.
# 4. Artifact Injection: 
#    - Copies custom .jsonl datasets into the vLLM benchmark directory.
#    - Validates the presence of required custom Python scripts (throughput, 
#      latency, compilation, etc.) before starting the build.
# 5. Post-Build Setup: Triggers a Python-based Hugging Face login utility 
#    using dynamic credentials for model access.
#################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CONFIG_FILE="config/user_config/static_config.json"
DYNAMIC_FILE="config/user_config/dynamic_config.json"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Custom vLLM Benchmark Setup{NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if config.json exists
if [ ! -f "${CONFIG_FILE}" ]; then
  echo -e "${RED}❌ Error: ${CONFIG_FILE} not found!${NC}"
  exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  echo -e "${YELLOW}⚠️  Warning: 'jq' is not installed. Installing jq...${NC}"
  if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y jq
  elif command -v yum &> /dev/null; then
    sudo yum install -y jq
  elif command -v brew &> /dev/null; then
    brew install jq
  else
    echo -e "${RED}❌ Error: Cannot install jq. Please install it manually.${NC}"
    exit 1
  fi
fi

# Read configuration from JSON
echo "Reading configuration from ${CONFIG_FILE}..."

COMMIT_ID=$(jq -r '.commit_id' "${CONFIG_FILE}")
FOLDER_PATH=$(jq -r '.filepath_of_the_project' "$CONFIG_FILE")
VENV_PATH=$(jq -r '.env_path' "$CONFIG_FILE")

# Validate required fields
if [ "${COMMIT_ID}" == "null" ] || [ -z "${COMMIT_ID}" ]; then
  echo -e "${RED}❌ Error: commit_id not found in ${CONFIG_FILE}${NC}"
  exit 1
fi

if [ "${FOLDER_PATH}" == "null" ] || [ -z "${FOLDER_PATH}" ]; then
  echo -e "${RED}❌ Error: filepath_of_the_project not found in ${CONFIG_FILE}${NC}"
  exit 1
fi

# Directory where the actual vLLM repository will be cloned
VLLM_REPO_PATH="$FOLDER_PATH/vllm"

if [[ ! -d "$VLLM_REPO_PATH" ]]; then
  echo "Cloning vLLM repository..."
  git clone https://github.com/vllm-project/vllm.git "$VLLM_REPO_PATH"
  cd "$VLLM_REPO_PATH"
  echo "Checking out specified vllm commit ID: $COMMIT_ID"
  git fetch origin
  git checkout -q "$COMMIT_ID"

else
  echo "vLLM repository already exists at $VLLM_REPO_PATH"
  # Always sync the state to the specific commit
  echo "Syncing vLLM repo to $COMMIT_ID"
  cd "$VLLM_REPO_PATH"
  git fetch origin
  git checkout -q "$COMMIT_ID"
fi

cd "$FOLDER_PATH"


SRC="custom_datasets"
DEST="vllm/benchmarks"

cp "$SRC"/*.jsonl "$DEST"
echo "COPIED CUSTOM DATASET FILES"

CUSTOM_FILES_DIR="custom-vllm-files"

echo -e "${GREEN}✅ Configuration loaded${NC}"
echo ""
echo -e "${BLUE}Build Configuration:${NC}"
echo ""

# Validate custom files exist
echo "Validating custom vLLM files..."
REQUIRED_FILES=(
  "throughput.py"
  "latency.py"
  "compilation.py"
  "envs.py"
  "run_batch.py"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "${CUSTOM_FILES_DIR}/${file}" ]; then
    MISSING_FILES+=("${file}")
  fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
  echo -e "${RED}❌ Error: Missing custom files in ${CUSTOM_FILES_DIR}/:${NC}"
  for file in "${MISSING_FILES[@]}"; do
    echo -e "${RED}  - ${file}${NC}"
  done
  echo ""
  echo "Please ensure all required files are in the ${CUSTOM_FILES_DIR}/ directory"
  exit 1
fi

echo -e "${GREEN}✅ All custom files found${NC}"
echo ""

# Show what will be included
echo "Custom files to be included:"
for file in "${REQUIRED_FILES[@]}"; do
  if [ -f "${CUSTOM_FILES_DIR}/${file}" ]; then
    FILE_SIZE=$(du -h "${CUSTOM_FILES_DIR}/${file}" | cut -f1)
    echo "  ✓ ${file} (${FILE_SIZE})"
  fi
done
echo ""

# Confirm before building
echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
read -r

cp -f "custom-vllm-files/run-performance-benchmarks.sh" "vllm/.buildkite/performance-benchmarks/scripts/run-performance-benchmarks.sh"
cp custom-vllm-files/throughput.py $VENV_PATH/vllm/benchmarks/throughput.py
cp custom-vllm-files/latency.py $VENV_PATH/vllm/benchmarks/latency.py
cp custom-vllm-files/compilation.py $VENV_PATH/vllm/config/compilation.py
cp custom-vllm-files/envs.py $VENV_PATH/vllm/envs.py
cp custom-vllm-files/run_batch.py $VENV_PATH/vllm/entrypoints/openai/run_batch.py


# Call Python script for hf access
python scripts/setup/hf_login.py --config_json_path $DYNAMIC_FILE

echo "Setup built successfully"
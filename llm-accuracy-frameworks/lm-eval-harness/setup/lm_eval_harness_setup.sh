# version-1 of the lm-eval-harness setup(not fully optimized)
#!/bin/bash

#################################################################################
# PURPOSE: Optimized LM-Evaluation-Harness Environment Setup
# -------------------------------------------------------------------------------
# This script automates the installation and configuration of the EleutherAI 
# LM Evaluation Harness. Key functionalities include:
# 
# 1. VERSION CONTROL: Performs a shallow, non-checkout clone followed by a 
#    targeted fetch of a specific Commit SHA to ensure evaluation reproducibility.
# 2. COMPUTE STACK: Installs a specific version of PyTorch (v2.8.0) compiled 
#    for CUDA 12.8 to leverage modern GPU acceleration.
# 3. HF ECOSYSTEM: Configures the core Hugging Face stack (Transformers, 
#    Accelerate, PEFT) with version-locked dependencies.
# 4. HARNESS INSTALLATION: Installs the 'lm-eval' package with specialized 
#    extras (math, ifeval, sentencepiece) required for complex benchmarking tasks.
#################################################################################

set -euo pipefail  # exit on error, undefined vars, pipe failures

# --- Configuration ---
REPO_URL=https://github.com/EleutherAI/lm-evaluation-harness.git
COMMIT_SHA=5d7dc4c0bbaac6f1186a8086bbb9c1da43be24cf
REPO_DIR=$(basename "$REPO_URL" .git)

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
# 1. Clone repository (minimal)
##############################################
print_status_message "Starting Clone Process"
git clone --no-checkout "$REPO_URL" "$REPO_DIR"

if [ $? -ne 0 ]; then
    print_status_message "Error: Clone failed"
    exit 1
fi

cd "$REPO_DIR"

##############################################
# 2. Fetch specific commit only
##############################################
print_status_message "Fetching commit $COMMIT_SHA"
git fetch --depth 1 origin "$COMMIT_SHA"

##############################################
# 3. Checkout commit
##############################################
print_status_message "Checking out commit $COMMIT_SHA"
git checkout "$COMMIT_SHA"

##############################################
# 4. Verify
##############################################
echo "Checked out commit:"
print_status_message "Clone Successful"


echo "Installing PyTorch (CUDA 12.8)..."
pip install --no-cache-dir \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "Installing Hugging Face core dependencies..."
pip install --no-cache-dir \
  transformers==4.57.2 \
  accelerate==1.12.0 \
  peft==0.18.0

echo "Installing lm-eval-harness with HF extras..."
pip install --no-cache-dir "lm_eval[hf]"

echo "Installing local package (editable)..."
pip install --no-cache-dir -e ".[math,ifeval,sentencepiece]"

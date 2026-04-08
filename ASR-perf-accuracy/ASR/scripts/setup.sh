#!/bin/bash

# ==============================================================================
# PURPOSE: Environment Setup, Dependency Installer & File Synchronizer
# ------------------------------------------------------------------------------
# This script initializes the workspace for the ASR Leaderboard by:
#   1. Installing Python dependencies for sub-modules and the root project.
#   2. Synchronizing/Replacing specific configuration or source files across 
#      the directory structure.
#   3. Ensuring all execution scripts (.sh) have 'execute' permissions.
#   4. Preparing the environment for a benchmarking run.
# ==============================================================================

OPEN_ASR_PATH="open_asr_leaderboard"
COMMIT_ID="b24cec5da30b0d58bc984b0201953b4fb70e9143"

if [[ ! -d "$OPEN_ASR_PATH" ]]; then
  echo "Cloning oepn_asr repository..."
  git clone https://github.com/huggingface/open_asr_leaderboard.git "$OPEN_ASR_PATH"
  cd "$OPEN_ASR_PATH"
  echo "Checking out specified open_asr commit ID: $COMMIT_ID"
  git fetch origin
  git checkout -q "$COMMIT_ID"

else
  echo "open_asr repository already exists at $OPEN_ASR_PATH"
  # Always sync the state to the specific commit
  echo "Syncing open_asr repo to $COMMIT_ID"
  cd "$OPEN_ASR_PATH"
  git fetch origin
  git checkout -q "$COMMIT_ID"
fi

# --- 1. INSTALL DEPENDENCIES ---

echo "Installing sub-module requirements..."
pip install -r requirements/requirements.txt

echo "Installing root requirements..."
cd ..
pip install -r requirements.txt

# Move back into the project folder for file operations
cd "$OPEN_AS_PATH" || exit

# --- 2. FILE REPLACEMENTS (Overwrite existing files) ---
echo "Replacing specific configuration files..."
# Example: Overwriting a specific python script with a modified version
cp -f "custom_files/eval_utils.py" "open_asr_leaderboard/normalizer/eval_utils.py"
cp -f "custom_files/nemo_eval/run_eval.py" "open_asr_leaderboard/nemo_asr/run_eval.py"
cp -f "custom_files/whisper_eval/run_eval.py" "open_asr_leaderboard/transformers/run_eval.py"

# --- 3. FILE DISTRIBUTION (Copying to new locations) ---
echo "Distributing new assets..."
# Example: Copying a new dataset list or utility
cp "custom_files/datasets.txt" "open_asr_leaderboard"
cp "custom_files/models.txt" "open_asr_leaderboard"
cp "custom_files/run_parakeet_updated.sh" "open_asr_leaderboard/nemo_asr"
cp "custom_files/run_salm_updated.sh" "open_asr_leaderboard/nemo_asr"
cp "custom_files/run_whisper_updated.sh" "open_asr_leaderboard/transformers"
cp "custom_files/master_launcher.sh" "open_asr_leaderboard"

cp -r post_processing_scripts open_asr_leaderboard

cd "$OPEN_ASR_PATH"

# --- 4. CONFIGURE PERMISSIONS ---
echo "Setting script permissions..."

# Main Launcher
chmod +x master_launcher.sh

# Transformer Scripts
if [ -f "transformers/run_whisper_updated.sh" ]; then
    chmod +x transformers/run_whisper_updated.sh
fi

# NeMo Scripts
[ -f "nemo_asr/run_salm_updated.sh" ] && chmod +x nemo_asr/run_salm_updated.sh
[ -f "nemo_asr/run_parakeet_updated.sh" ] && chmod +x nemo_asr/run_parakeet_updated.sh

echo "✅ Setup and File Sync complete. Ready for benchmarking."
#!/bin/bash
# ==============================================================================
# PURPOSE: ASR Leaderboard Entry Point Wrapper
# ------------------------------------------------------------------------------
# This script acts as a safety wrapper to ensure the user is in the correct 
# workspace before launching the benchmarking suite.
#
# Key Functions:
#   1. Validates the current working directory.
#   2. Automatically navigates into the 'open_asr_leaderboard' directory 
#      if the user is currently in the parent folder.
#   3. Triggers the 'master_launcher.sh' orchestration script.
# ==============================================================================

OPEN_ASR_PATH="open_asr_leaderboard"

# Check if already inside the folder
if [[ "$(basename "$PWD")" != "$OPEN_ASR_PATH" ]]; then
    cd $OPEN_ASR_PATH
    echo "Directory changed"
fi

./master_launcher.sh
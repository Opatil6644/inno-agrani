OPEN_ASR_PATH="vllm-omni"

# Check if already inside the folder
if [[ "$(basename "$PWD")" != "$OPEN_ASR_PATH" ]]; then
    cd $OPEN_ASR_PATH
    echo "Directory changed"
fi

./master_launcher.sh
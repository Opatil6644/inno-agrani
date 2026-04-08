#!/bin/bash

# Get the absolute path of the current directory
BASE_DIR=$(pwd)
echo "Starting from: $BASE_DIR"

# Path to your model list
MODEL_LIST="master_model_list.txt"

# Ensure the list file exists
if [[ ! -f "$MODEL_LIST" ]]; then
    echo "Error: $MODEL_LIST not found."
    exit 1
fi

echo "Starting automated profiling loop..."

while IFS= read -r line || [ -n "$line" ]; do
    # 1. Skip lines that start with '#' (ignoring leading whitespace)
    if [[ "$line" =~ ^[[:space:]]*# ]]; then
        # echo "Skipping commented model: $line"
        continue
    fi

    # 2. Strip whitespace and hidden carriage returns (\r) for active models
    clean_line=$(echo "$line" | tr -d '\r' | xargs)
    
    # Skip truly empty lines
    [ -z "$clean_line" ] && continue

    echo "------------------------------------------------"
    echo "Processing Tag: $clean_line"
    echo "------------------------------------------------"

    # 3. Mapping keywords to their specific directories
    case $clean_line in
        "helios")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/helios"
            ;;
        "image_to_image")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/image_to_image"
            ;;
        "image_to_video")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/image_to_video"
            ;;
        "text_to_image")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/text_to_image"
            ;;
        "text_to_audio")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/text_to_audio"
            ;;

        "fishaudio")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/fish_speech"
            ;;

        "hunyuanImage3.0")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/hunyuan_image3"
            ;;

        "text_to_video")
            TARGET_DIR="$BASE_DIR/examples/offline_inference/text_to_video"
            ;;
        *)
            echo "Skipping unknown tag: $clean_line"
            continue
            ;;
    esac

    # 4. Navigation and Execution Block
    if [ -d "$TARGET_DIR" ]; then
        pushd "$TARGET_DIR" > /dev/null
        
        if [ -f "run.sh" ]; then
            echo "Executing run.sh in $TARGET_DIR..."
            bash run.sh
        else
            echo "Error: run.sh not found in $TARGET_DIR"
        fi

        popd > /dev/null
    else
        echo "Error: Directory not found!"
        echo "Looked in: $TARGET_DIR"
    fi

done < "$MODEL_LIST"

echo "------------------------------------------------"
echo "All profiling tasks finished."
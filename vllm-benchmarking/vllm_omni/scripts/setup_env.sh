#!/bin/bash
set -e  # Exit on error

echo "=== Starting Environment Setup for B200/vLLM-Omni ==="

COMMIT_ID="92f771d45507b95a627c566268a924140b3e7c05"

# 1. Installing uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv is already installed."
fi

# 2. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with Python 3.12..."
    uv venv --python 3.12 --seed
else
    echo "Virtual environment already exists."
fi

# 3. Activate Environment
source .venv/bin/activate

# 4. Install vLLM (Specific version requested)
echo "Installing vLLM v0.17.0..."
uv pip install vllm==0.17.0 --torch-backend=auto
uv pip install openpyxl==3.1.5

uv pip install fish-speech

# 5. Handle vllm-omni repo
# If we are already inside a vllm-omni directory, just install. 
# Otherwise, clone it.
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "Already inside vllm-omni directory. Installing in editable mode..."
    uv pip install -e .
else
    echo "Cloning vllm-omni..."
    git clone https://github.com/vllm-project/vllm-omni.git
    cd vllm-omni
    echo "Checking out specified vllm-omni commit ID: $COMMIT_ID"
    git fetch origin
    git checkout -q "$COMMIT_ID"
    uv pip install -e .
fi

echo "=== Setup Complete! ==="
echo "To activate manually: source .venv/bin/activate"

cd ..

# Custome files are copied inside vllm-omni repo
cp "custom-vllm-files/cherry_blossom.jpg" "vllm-omni/examples/offline_inference/image_to_video"
cp "custom-vllm-files/master_launcher.sh" "vllm-omni"
cp "custom-vllm-files/master_model_list.txt" "vllm-omni"
cp "custom-vllm-files/qwen-bear.png" "vllm-omni/examples/offline_inference/image_to_image"
cp "custom-vllm-files/run_fish_speech.sh" "vllm-omni/examples/offline_inference/fish_speech/run.sh"
cp "custom-vllm-files/run_helios.sh" "vllm-omni/examples/offline_inference/helios/run.sh"
cp "custom-vllm-files/run_image_to_image.sh" "vllm-omni/examples/offline_inference/image_to_image/run.sh"
cp "custom-vllm-files/run_image_to_video.sh" "vllm-omni/examples/offline_inference/image_to_video/run.sh"
cp "custom-vllm-files/run_text_to_audio.sh" "vllm-omni/examples/offline_inference/text_to_audio/run.sh"
cp "custom-vllm-files/run_text_to_image.sh" "vllm-omni/examples/offline_inference/text_to_image/run.sh"
cp "custom-vllm-files/run_text_to_video.sh" "vllm-omni/examples/offline_inference/text_to_video/run.sh"
cp "custom-vllm-files/cherry_blossom.jpg" "vllm-omni/examples/offline_inference/hunyuan_image3"
cp "custom-vllm-files/run_hunyuan.sh" "vllm-omni/examples/offline_inference/hunyuan_image3/run.sh"

cp "custom-vllm-files/hunyuan_image_3_moe.yaml" "vllm-omni/vllm_omni/model_executor/stage_configs/hunyuan_image_3_moe.yaml"

cp -r post_processing_scripts vllm-omni
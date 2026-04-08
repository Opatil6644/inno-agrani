# MODEL ARCHITECTURE HTML REPORT
This setup is a comprehensive LLM Architecture Profiler. It visualizes the internal structure of a Hugging Face model by generating an interactive HTML tree that displays parameter counts, memory usage (CPU/GPU), and the exact tensor shapes flowing through every layer during a forward pass.

## Prerequisites
Ensure your system meets the following requirements before proceeding:

- Operating System: Linux
- Shell: Bash
- Hardware: CUDA-compatible GPU
- Python 3.13(Recommended)
- Docker
- jq (version 1.7.1)
- Git
- yq (version 3.4.3)
- torch
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup

### 1. Environment Preparation
It is highly recommended to use a virtual environment to manage dependencies.

### 2. Clone this repository
```
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/vllm-benchmarking.git
cd your-repo-directory
cd html_model_report
```

### 3. Install Dependencies
This setup utilizes a requirements.txt file.
```bash
pip install -r requirements.txt
```

### 4. Static Configuration
Before building the environment, define your hf token configuration file.
- Open config/config.json.
- Set the hf_token.

### 5. Model Selection
The pipeline only generates html report for models that are explicitly enabled.

### Instructions for Configuring models.yaml
#### 1. Activate Models:
- Go to the models: section and uncomment (remove the #) the specific models you wish to evaluate.

### 2. Execute Benchmark
#### Ensure the script has execution permissions(One time process)
```bash
chmod +x evaluate/eval.sh
```

#### Run the main evaluation script.
```bash
evaluate/eval.sh
```
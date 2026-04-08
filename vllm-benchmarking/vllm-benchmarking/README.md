# VLLM-BENCHMARKING FRAMEWORK
The vLLM Benchmarking Framework is a specialized diagnostic ecosystem designed to quantify the performance limits and operational efficiency of Large Language Models (LLMs) powered by the vLLM inference engine. By decoupling the complexities of environment configuration from the evaluation process, it provides a standardized "laboratory" for comparative model analysis.

## Prerequisites
Ensure your system meets the following requirements before proceeding:

- Operating System: Linux
- Shell: Bash
- Hardware: CUDA-compatible GPU
- jq (version 1.7.1)
- Git
- Python(3.12 recommended)
- yq (version 3.4.3)
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup

### 1. Environment Preparation
It is highly recommended to use a virtual environment to manage dependencies.
```
python3 -m venv vllm_env
```
- Activate the venv
```
source vllm_env/bin/activate
```

### 2. Clone this repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/vllm-benchmarking.git
cd your-repo-directory
cd vllm-benchmarking
```

### 3. Install Dependencies
This setup utilizes a requirements.txt file.
- 1.Minimal requirements
```bash
pip install -r requirements.txt
```
- 2. Full requirments.txt(Use this if any issues faced during minimal requirements.txt)
```bash
pip install -r full_requirements.txt
```

### 4. Static Configuration
Before building the environment, define your project root path in the static configuration file.

- Open config/user_config/static_config.json.
- Set the filepath_of_the_project to your absolute directory path.
- Example: "/root/vllm-benchmarking/vllm-benchmarking"
- env_path: Paste the full path of the python venv created(example: /root/vllm_env/lib/python3.12/site-packages)

### 5. Dynamic Configuration
Configure the execution parameters in config/user_config/dynamic_config.json. Populate the following fields:

| Field                | Description |
|----------------------|-------------|
| `huggingface_token`  | Your HF access token for gated models. |
| `enforce_eager`      | Set to `true` or `false` to toggle PyTorch eager mode. |
| `VLLM_ATTENTION_BACKEND`| Specify the attention backend used by vLLM. |
| `VLLM_COMPILATION_MODE` | Define the specific vLLM compilation optimization level. |

Note: Use TRITON_ATTN as VLLM_ATTENTION_BACKEND when running gpt-oss-20b model.

### 6. Build Custom Setup (One-Time Process)
Initialize the environment by building the custom benchmarking setup.
```bash
chmod +x scripts/setup/build_setup.sh
scripts/setup/build_setup.sh
```

## Evaluation & Benchmarking

### 1. Model Selection
The pipeline only evaluates models that are explicitly enabled.

#### Instructions for Configuring models.yaml
To correctly set up your evaluation run, please follow these steps:

#### 1. Configure Model Overrides:
- By default, the system uses configurations from config/model_details/details.csv.
- If you need to override settings like tensor_parallel size or trust_remote_code, create a custom CSV file following the template in the comments.
- Provide the full path to your CSV file in the path: custom_config_path: field.

#### 2. Choose Dataset Mode
You must choose between Default and Custom modes:

##### Option A: Default Mode
- Set mode: default.
- The system will run a standard suite of datasets using pre-defined benchmark settings for num_prompts and output_length.

##### Option B: Custom Mode
- Set mode: custom.
- Select Datasets: Look at the available list for valid options. Under the selected list, uncomment only the datasets you wish to run.
- Set Constraints: In the evaluation section, define your own num_prompts (total prompts to run) and output_length (max tokens per response). These values are only applied when the mode is set to custom.

#### 3. Activate Models:
- Go to the models: section and uncomment (remove the #) the specific models you wish to evaluate.

### 2. Execute Benchmark
#### Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/evaluate/run-benchmark.sh
```

#### Run the main evaluation script.
```bash
scripts/evaluate/run-benchmark.sh
```

Note:
- Results folder will be automatically created at this path `<filepath_of_the_project>` so don't manually create one.
- Some HuggingFace models require file access, please get the file access in that case.

### Results Directory Structure
```text
Results/
├── model_name/
│   ├──Custom_dataset
|      ├── BBH
│          ├── perf_benchmark/
│          │   ├── latency_results.json
|          |   ├── throughput_results.json
|          |   ├── serving_results.json*
|          ├── cublas_func_calls.csv
|          ├── kernel_summary.csv
|          ├── mnk_details_sorted.csv
|          ├── mnk_details.csv
|          ├── nvjet_details.xlsx
|          ├── open_closed_source_kernels.xlsx
│   ├──Default_dataset
│          ├── perf_benchmark/
│          │   ├── latency_results.json
|          |   ├── throughput_results.json
|          |   ├── serving_results.json*
|          ├── benchmark.log
|          ├── cublas_func_calls.csv
|          ├── kernel_summary.csv
|          ├── mnk_details_sorted.csv
|          ├── mnk_details.csv
|          ├── nvjet_details.xlsx
|          ├── open_closed_source_kernels.xlsx
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx
```

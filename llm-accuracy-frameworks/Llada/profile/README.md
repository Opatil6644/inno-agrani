# LLaDA Model Inference & Profiling Pipeline
This pipeline is designed to profile llada model using custom(offical) llada codebase.

# Prerequisites
Ensure your environment meets the following hardware and software requirements:

- Python: 3.13 (Recommended)
- CUDA(Recommended for GPU acceleration)
- PyTorch 2.8.0
- Operating System: Linux
- Shell: Bash
- jq (version 1.7.1)
- yq
- Git
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup
## 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.

## 2. Clone the Repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/llm-accuracy-frameworks.git
cd <repository-directory>
cd Llada
```

## 3. Install Dependencies
This setup utilizes a requirements.txt file.
```bash
pip install -r profile/requirements.txt
```

## 4. Setup & Configuration
Configure User Settings:
Edit profile/config/user_eval_setting.json to define your results path.

## Step 5. Evaluate
Run the main evaluation script to begin the profiling and evaluation process

Ensure the script has execution permissions(One time process)
```bash
chmod +x profile/llada_eval.sh
profile/llada_eval.sh
```


### Results Directory Structure
```text
Results/
├── model_name/
|   ├── cublas_func_calls.csv
|   ├── kernel_summary.csv
|   ├── mnk_details_sorted.csv
|   ├── mnk_details.csv
|   ├── nvjet_details.xlsx
|   ├── open_closed_source_kernels.xlsx
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx

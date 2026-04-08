# LLM Inference Profiler & Kernel Analyzer using TensorRT
This repository contains a benchmarking and profiling suite designed to evaluate LLM performance across different torch.compile backends. It combines high-level performance metrics (Latency, Throughput) with deep-dive GPU kernel analysis using NVIDIA Nsys.

## Prerequisites
Ensure your environment meets the following hardware and software requirements:

- Python: 3.12 (Recommended)
- CUDA:(Recommended for high-performance VRAM management)
- Operating System: Linux
- Shell: Bash
- jq
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup
### 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.

### 2. Clone the Repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/vllm-benchmarking.git
cd <repository-directory>
cd tensorrt_setup
```

### 3. Install Dependencies
This setup utilizes a requirements.txt file.
```bash
pip install -r requirements.txt
```

### 4. Configuration
Before running the pipeline, you must configure few details.

Navigate to config/user_evaluation_settings.json.
- Update the results_path field with your desired directory.
- Update your HF access token for gated models.

### 5. Setup (One-Time Process)
Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/evaluate/tensorrt_model_eval.sh
```

### 6. Running Evaluation
#### Step 1. Model Selection
- Open master_model_list.txt and uncomment the generative models you wish to profile
- Open backed_list.txt and uncomment the backend combination you wish to profile.

#### Step 2. Evaluate
Run the main evaluation script to begin the profiling and evaluation process
```bash
scripts/evaluate/tensorrt_model_eval.sh
```

Note: **If you face any issues while running the benchmarks please make sure LD_LIBRARY_PATH is configured.**

## Results Directory Structure
```text
Results/
├── model_name/
|   |── Profile_Details
│       ├── backend(inductor)/
|       |       ├── cublas_func.csv
|       |       ├── kernel_summary.csv
|       |       ├── mnk_sorted.csv
|       |       ├── mnk.csv
|       |       ├── nvjet_details.xlsx
|       |       ├── open_closed_source_kernels.xlsx
|       |       ├── benchmark_backed_details.csv
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx
```


# Wan2.1 Video Generation & GPU Profiling Pipeline
This repository contains an automated pipeline for generating high-fidelity videos using the Wan2.1-T2V-14B model while simultaneously performing deep GPU architectural profiling.

## Prerequisites
Ensure your environment meets the following hardware and software requirements:
- Python: 3.13 (Recommended)
- Operating System: Linux
- Shell: Bash
- jq (version 1.7.1)
- Git
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup
## 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.

## 2. Clone the Repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/videogen-perf-accuracy.git
cd <repository-directory>
```

## 3. Install Dependencies
This setup utilizes a requirements.txt file.
```bash
pip install -r requirements.txt
```

## 4. Configuration
Before running the pipeline, you must define where your results will be stored:

Navigate to config/user_eval_setting.json.
- Update the results_path field with your desired directory.

## Running Evaluation
## 1. Set Permissions
Ensure the evaluation script has execution permissions (one-time process)
```bash
chmod +x model_eval.sh
```

## 2. Execute
Run the main evaluation script to begin the profiling and evaluation process
```bash
./model_eval.sh
```

### Results Directory Structure
```text
Results/
├── model_name/
|   ├── Profile_Details
|   ├───├── cublas_func_calls.csv
|   ├───├── kernel_summary.csv
|   ├───├── mnk_details_sorted.csv
|   ├───├── mnk_details.csv
|   ├───├── nvjet_details.xlsx
|   ├───├── open_closed_source_kernels.xlsx
|   ├───├── output.mp4

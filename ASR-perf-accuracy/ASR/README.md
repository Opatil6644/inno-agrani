# ASR Master Leaderboard Runner
This repository provides an automated pipeline to benchmark multiple ASR models against various datasets. It handles directory routing, model-specific execution, and centralized post-processing to generate a final leaderboard.

## Prerequisites
Ensure your environment meets the following hardware and software requirements:

- Python: 3.12.3 (Recommended)
- CUDA:(Recommended for high-performance VRAM management)
- Operating System: Linux
- Shell: Bash
- jq (version 1.7.1)
- Git
- pip
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup
## 1. Environment Setup
It is highly recommended to use a virtual environment to avoid dependency conflicts.

## 2. Clone the Repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/ASR-perf-accuracy.git
cd <repository-directory>
cd ASR
```

## 3. Configuration
Before running the pipeline, you must define your hf token,

Navigate to .env file
- Update your HF access token for gated models.

## 4. Prepare setup script (One-Time Process)
Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/setup.sh
scripts/setup.sh
```
## 5. Running Evaluation
### Step 1. Model Selection
Open ASR/open_asr_leaderboard/model_list.txt and uncomment the generative models you wish to profile

### Step 2. Dataset Selection
Open ASR/open_asr_leaderboard/dataset.txt and uncomment the dataset you wish to profile, **please run the evaluation selecting one dataset at a time**.

### Step 3. Evaluate
Run the main evaluation script to begin the profiling and evaluation process

Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/evaluate.sh
scripts/evaluate.sh
```

### Results Directory Structure
By default Results will be stored here **ASR/open_asr_leaderboard/Results**

```text
Results/
├── model_name/
│   ├── Profile_Details/
│   ├───├──Dataset_name/
|   ├────────├── cublas_func_calls.csv
|   |────────├── kernel_summary.csv
|   |────────├── mnk_details_sorted.csv
|   |────────├── mnk_details.csv
|   |────────├── nvjet_details.xlsx
|   |────────├── open_closed_source_kernels.xlsx
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx
|__ final_leaderboard_summary.xlsx
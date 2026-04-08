# Text-to-Image Automation: Evaluation(evalGIM) & Profiling
This repository provides an automated pipeline for profiling and evaluating Text-to-Image models using EvalGIM library. This setup will run evaluations with the full COCO validation dataset when computing marginal metrics like FID and precision, recall, coverage, and diversity.

## Prerequisites
Ensure your environment meets the following hardware and software requirements:

- Python: 3.10 (Recommended)
- CUDA:(Recommended for high-performance VRAM management)
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
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/evalgim-harness.git
cd <repository-directory>
```

## 3. Configuration
Before running the pipeline, you must configure few details.

- Navigate to config/user_evaluation_settings.json.
- Update your HF access token for gated models.
- Update coco_data_path where you want to download the coco dataset.
- Update the results_path field with your desired directory.
- Update num_samples, i.e number of images you want to generate(recommended to set 5000 for better results).
- Update batch_size, batch size used during image generation.
- Update save_generated_images, i.e whether to save the generated images(recommended to be false to reduce memory overhead)

## 4. Prepare evaluation configuration (One-Time Process)
Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/setup/prepare_evaluation_config.sh
scripts/setup/prepare_evaluation_config.sh
```

## 5. Setup (One-Time Process)
Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/setup/setup.sh
scripts/setup/setup.sh
```
## 6. Running Evaluation
### Step 1. Model Selection
Open master_model_list.txt and uncomment the generative models you wish to profile

### Step 2. Evaluate
Run the main evaluation script to begin the profiling and evaluation process

Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/evaluate/model_evaluation.sh
scripts/evaluate/model_evaluation.sh
```

### Option 1
Uses defualt master_model_list.txt
```bash
evaluate/model_eval.sh
```
### Option 2
Custom path to master_model_list.txt.
```bash
evaluate/model_eval.sh <custom_models.txt>
```

## Supported metrics
The following is the current set of supported metrics:

## Marginal Metrics:
- FID (Fréchet Inception Distance)
- Precision/recall/diversity/coverage

## Conditional Metics:
- CLIPScore (torchmetrics implementation)

### Results Directory Structure
```text
Results/
├── model_name/
│   ├── Evaluation/
│   │       ├── scores.yaml
│   ├── Profile_Details/
|   |       ├── cublas_func_calls.csv
|   |       ├── kernel_summary.csv
|   |       ├── mnk_details_sorted.csv
|   |       ├── mnk_details.csv
|   |       ├── nvjet_details.xlsx
|   |       ├── open_closed_source_kernels.xlsx
│   ├── latency.json
│   ├── throughput.json
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx
|__ latency_throughput_summary.csv
|__ results.csv

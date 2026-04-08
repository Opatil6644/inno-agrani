# Vision Transformer (ViT) Automation: Evaluation & Profiling
The Vision Transformer (ViT) Automation Framework is a specialized diagnostic suite designed to bridge the gap between computer vision and high-performance transformer execution. 

## Prerequisites
Ensure your environment meets the following hardware and software requirements

- Python: 3.13 (Recommended)
- CUDA(Recommended for GPU acceleration)
- PyTorch 2.8.0
- Operating System: Linux
- Shell: Bash
- jq (version 1.7.1)
- Git
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

## Installation & Setup
## 1. Environment Setup
It is highly recommended to use a virtual environment

## 2. Clone the Repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/vision-transformer-benchmarking.git
cd <repository-directory>
```

## 3. Install Dependencies
This setup utilizes a requirements.txt file. Install the specific versions of Torch followed by the remaining dependencies:
```bash
pip install -r requirements.txt
```

## 4. Configuration
Navigate to config/user_config.json.

- Update the results_path field to your desired output directory.
- Update the img_path if you want to use your own custom image, if not default image would be - considered(from config folder).
- Update your HF access token for gated models.

## Running Evaluation
The pipeline processes only the models enabled in your master list.

## 1. Set Permissions
Ensure the evaluation script has execution permissions (one-time process).
```bash
chmod +x evaluate/model_eval.sh
```

## Step 2. Model Selection
Open master_model_list.txt and uncomment (remove the #) the ViT models you wish to evaluate.

## Step 3. Execute
Run the automation script.
## Option 1
Uses defualt master_model_list.txt
```bash
evaluate/model_eval.sh
```
## Option 2:
Custom path to master_model_list.txt.
```bash
evaluate/model_eval.sh <custom_models.txt>
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

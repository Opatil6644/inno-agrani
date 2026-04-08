# BERT Evaluation & Profiling Pipeline
This framework is designed to profile various bert based models.

## Prerequisites
Ensure your environment meets the following hardware and software requirements:

- Python: 3.13 (Recommended)
- CUDA(Recommended for GPU acceleration)
- PyTorch 2.8.0
- Transformers 4.57.3
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
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/nlp-benchmarking.git
cd <repository-directory>
cd BERT_EVAL
```

## 3. Install Dependencies
This setup utilizes a requirements.txt file.
```bash
pip install -r requirements.txt
```

## 4. Configuration
Before running the pipeline, you must define where your results will be stored:

Navigate to config/user_config.json.
- Update the results_path field with your desired directory.
- Update hf_token.

## Running Evaluation
The pipeline uses a selective execution method. Only models explicitly "uncommented" in the master list will be processed. 

## 1. Model Selection
Open models.yaml and uncomment the models and type you wish to evaluate.

## 2. Set Permissions
Ensure the evaluation script has execution permissions(one-time process)
```bash
chmod +x evaluate/model_eval.sh
```

## 3. Execute
Run the main evaluation script to begin the profiling and evaluation process
```bash
evaluate/model_eval.sh
```

### Results Directory Structure
```text
Results/
├── model_name/
|   ├── Eval_Details
|   ├───├── results.json
|   ├── Profile_Details
|   ├───├── cublas_func_calls.csv
|   ├───├── kernel_summary.csv
|   ├───├── mnk_details_sorted.csv
|   ├───├── mnk_details.csv
|   ├───├── nvjet_details.xlsx
|   ├───├── open_closed_source_kernels.xlsx
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx

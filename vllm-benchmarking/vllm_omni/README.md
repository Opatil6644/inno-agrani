# vLLM-Omni: Multimodal Inference & Profiling Suite
A unified inference engine built on vLLM for end-to-end multimodal generation. This setup integrates state-of-the-art models for vision, video, audio, and text, featuring a deep-profiling layer to analyze GPU kernel performance.

### 1. Clone this repository
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/vllm-benchmarking.git
cd your-repo-directory
cd vllm_omni
```

## 2. Prepare setup script (One-Time Process)
Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/setup_env.sh
scripts/setup_env.sh
```

## 3. Activate the venv
```
source .venv/bin/activate
```

## Optional
If you face any package/library issues while running the above setup_env.sh, use the below full requirements.txt file
```
pip install -r full_requirements.txt
```

## Running Evaluation
### Step 1. Model Selection
Open vllm_omni/vllm-omni/master_model_list.txt and uncomment the models you wish to profile.

### Step 2. Evaluate
Run the main evaluation script to begin the profiling and evaluation process

Ensure the script has execution permissions(One time process)
```bash
chmod +x scripts/evaluate.sh
scripts/evaluate.sh
```

### Results Directory Structure
By default Results will be stored here **vllm_omni/vllm-omni/Results**

```text
Results/
├── model_name/
|   ├── cublas_func_calls.csv
|   ├── kernel_summary.csv
|   ├── mnk_sorted.csv
|   ├── mnk.csv
|   ├── nvjet_details.xlsx
|   ├── open_closed_source_kernels.xlsx
|__ mnk_aggregated.csv
|__ nvjet_aggregated.xlsx
|__ open_close_source_kernels_aggregate.xlsx

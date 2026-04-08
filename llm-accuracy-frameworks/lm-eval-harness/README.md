# Language Model Evaluation Harness
This pipeline is designed to evaluate llm's using EleutherAI/lm-evaluation-harness library. The evaluation is performed on standard datasets like BBH, GPQA, IFEVAL, MATH, MMLU and MUSR.

## Prerequisites

Before starting, ensure the following are installed:

- Python: 3.10(recommended)
- Git
- CUDA-compatible GPU (recommended for large models)
- pip
- OS - Linux
- Bash shell
- Nsight Systems version (tested with this version 2024.6.2.225-246235244400v0)

---

## Installation

### Step 1: Create Virtual Environment (Highly Recommended)

### Step 2: Download Setup Script(Clone this repository)
```bash
git clone https://github.com/AGL-Innominds-Libraries-and-Tools/llm-accuracy-frameworks.git
cd your-repo-directory
cd lm-eval-harness
```

### Step 3: Make Script Executable
```bash
chmod +x setup/lm_eval_harness_setup.sh
```

### Step 4: Run Setup Script (One-Time Process)
```bash
setup/lm_eval_harness_setup.sh
```

### Usage
Running Evaluations

### Basic evaluation command structure:
Uncomment/Add the models you wish to evaluate in models.yaml before running the command.

### Option 1: Below command uses default models.yaml
```bash
python evaluate/run_eval.py <leaderboard_task> <path_to_store_results>
```

### Option 2: Custom path to models.yaml.
```bash
python evaluate/run_eval.py <leaderboard_task> <path_to_store_results> --config <custom_models.yaml_path>
```

Note:
- Results folder will be automatically created at this path `<path_to_store_results>` so don't manually create one.
- Some HuggingFace models require executing custom repository code that expects `trust_remote_code=True`.
If you encounter this error: Please pass the argument `trust_remote_code` to allow custom code to be run. Please use the below command with additional argument --trust-remote-code.
```bash
python evaluate/run_eval.py <leaderboard_task> <path_to_store_results> --trust-remote-code
```



### Available Tasks
- leaderboard_bbh
- leaderboard_gpqa
- leaderboard_ifeval
- leaderboard_math_hard
- leaderboard_mmlu_pro
- leaderboard_musr
- leaderboard (Runs all available tasks. Note: Not recommended for single runs as it is time-consuming.)

### Results Directory Structure
```text
results/
├── model_name/
│   ├── raw_results/
│   │   ├── *.json
│   ├── normalized_results/
│   │   ├── *.json
|__ aggregated_results.csv
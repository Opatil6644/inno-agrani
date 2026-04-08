# LLaDA Evaluation Setup
This pipeline is designed to evaluate llada model using offical llada codebase.

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
pip install -r eval/requirements.txt
```

## 4. Evaluate
To evaluate the model on specific benchmarks, follow these steps:
- Select your task: Open eval_llada_lm_eval.sh and locate the benchmark you wish to run (e.g., MMLU, GSM8K, or ARC).
- Enable the command: Uncomment the corresponding accelerate launch line by removing the # at the start of the line.

Ensure the script has execution permissions(One time process)
```bash
chmod +x eval/eval_llada_lm_eval.sh
eval/eval_llada_lm_eval.sh
```
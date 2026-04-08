# Environment Setup (lm_eval_harness_setup.sh)
To ensure reproducible results, this setup uses a specific version of the lm-evaluation-harness and optimized CUDA-enabled dependencies.

### Features
- Version Pinning: Clones the EleutherAI/lm-evaluation-harness repository and checks out a specific, tested commit.

- Optimized Compute: Installs PyTorch 2.8.0 with CUDA 12.8 support.

- HF Ecosystem: Installs compatible versions of transformers, accelerate, and peft.

- Task-Specific Dependencies: Installs the local package in editable mode with extras required for math, ifeval, and sentencepiece.
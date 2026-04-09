# Setup Overview: ASR Master Leaderboard & Profiler
## 1. Project Overview
The ASR Master Leaderboard Runner is an end-to-end automation pipeline designed to evaluate ASR models (like Whisper, Nvidia-Canary, etc.) against standardized datasets. It orchestrates the entire lifecycle: from environment setup and model weight retrieval to GPU kernel profiling and the generation of a accuracy leaderboard.

## 2. Key Differentiation
This repository is optimized for cross-dataset validation. It allows developers to see how a model's hardware efficiency (profiling) changes when processing different acoustic environments or languages provided by various datasets.

## 3. Workflow & Orchestration
The pipeline uses a "Selective Execution" logic via text-file registries:
- Model Registry (model_list.txt): Defines the scope of the benchmark.
- Dataset Registry (dataset.txt): Controls the input data.
- Critical Note: The system is optimized for single-dataset runs. To ensure clean profiling data, it is recommended to uncomment only one dataset at a time.
- Environment Control: Uses a .env file for HuggingFace authentication, keeping sensitive tokens separate from the logic scripts.

## 4. Execution Logic
- Permissioning: Granting execution rights to the scripts/ directory.
- Initialization: setup.sh configures the local environment and ensures jq and Nsight Systems are ready.
- The Runner: evaluate.sh iterates through the enabled models in the list, performs the ASR transcription, and hooks the process into NVIDIA Nsight for hardware telemetry.

## 5. Result Analysis & Hierarchy
- cublas_func_calls.csv
Contains logs of all cuBLAS API function calls made during execution.

- kernel_summary.csv
High-level summary of all GPU kernels executed.

- mnk_details.csv
Raw data of matrix multiplication workloads in terms of:
M, N, K dimensions

- mnk_details_sorted.csv
Same as mnk_details.csv but:
Sorted (usually by frequency or compute cost)

- mnk_aggregated.csv
Aggregated version of MNK data:
Groups identical (M, N, K) combinations

- nvjet_details.xlsx
Detailed report generated using NVIDIA profiling tools.
Contains:
Kernel-level breakdown, shapes etc

- nvjet_aggregated.xlsx
Aggregated version of NVJET data:

- open_closed_source_kernels.xlsx
Classification of kernels into:
Open-source kernels 
Closed-source kernels

- open_close_source_kernels_aggregate.xlsx
Aggregated summary of above classification:


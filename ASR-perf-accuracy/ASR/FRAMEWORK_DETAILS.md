# Setup Overview: ASR Master Leaderboard & Profiler
## 1. Project Overview
The ASR Master Leaderboard Runner is an end-to-end automation pipeline designed to evaluate ASR models (like Whisper, Nvidia-Canary, etc.) against standardized datasets. It orchestrates the entire lifecycle: from environment setup and model weight retrieval to GPU kernel profiling and the generation of a final performance-vs-accuracy leaderboard.

## 2. Key Differentiation
Unlike the TTS or EvalGIM suites, this repository is optimized for cross-dataset validation. It allows developers to see how a model's hardware efficiency (profiling) changes when processing different acoustic environments or languages provided by various datasets.

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
The results are structured to provide both "Micro" (per model/dataset) and "Macro" (leaderboard) views.

### The Macro View (Root Results)
- final_leaderboard_summary.xlsx: The crown jewel of this repo. It aggregates the word error rates (WER/Accuracy) and performance metrics into a single comparative sheet.
- mnk_aggregated.csv: Compares the computational "shapes" of different ASR models to identify which ones are most mathematically efficient.
- The Micro View (Profile_Details/Dataset_name/)
For every model/dataset combination, the system captures:
- Kernel Summaries: identifies time spent in specific CUDA kernels.
- cuBLAS Calls: Monitors the linear algebra backbone of the Transformer layers.
- NVJet Details: High-resolution telemetry for power and thermal performance during transcription.
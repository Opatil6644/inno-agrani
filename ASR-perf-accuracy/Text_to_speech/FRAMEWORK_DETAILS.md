# Setup Overview: TTS Performance & Profiling Pipeline
## 1. Executive Summary
The AI Model Performance Profiling & TTS Suite is an automated framework designed to benchmark Text-to-Speech models. Unlike standard accuracy tests, this suite focuses on hardware-level profiling, capturing how TTS models interact with GPU kernels, memory, and CUDA libraries (cuBLAS) during the audio synthesis process.

## 2. Core Capabilities
- NVIDIA Nsight Integration: Leverages Nsight Systems to hook into the execution timeline of the model.
- Kernel Analysis: Distinguishes between proprietary (closed-source) and standard (open-source) GPU kernels to identify optimization barriers.
- Matrix Math Profiling: Extracts MNK dimensions (Matrix Multiply-Accumulate operations) to analyze Tensor Core utilization.

## Performance Profiling Outputs
- Unlike standard evaluation scripts, this setup provides deep-dive hardware profiling. All results are stored in the Profile_Details/ subdirectory for each model:
- Kernel Summaries: CSV files detail which GPU kernels (Open vs. Closed source) were utilized most frequently.
- MNK Details: Provides matrix multiplication dimensions (M, N, K) which are critical for optimizing tensor core utilization.

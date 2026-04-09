# Setup Overview: TTS Performance & Profiling Pipeline
## 1. Executive Summary
The AI Model Performance Profiling & TTS Suite is an automated framework designed to benchmark Text-to-Speech models. Unlike standard accuracy tests, this suite focuses on hardware-level profiling, capturing how TTS models interact with GPU kernels, memory, and CUDA libraries (cuBLAS) during the audio synthesis process.

## 2. Core Capabilities
- NVIDIA Nsight Integration: Leverages Nsight Systems to hook into the execution timeline of the model.
- Kernel Analysis: Distinguishes between proprietary (closed-source) and standard (open-source) GPU kernels to identify optimization barriers.
- Matrix Math Profiling: Extracts MNK dimensions (Matrix Multiply-Accumulate operations) to analyze Tensor Core utilization.

## Performance Profiling Outputs
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

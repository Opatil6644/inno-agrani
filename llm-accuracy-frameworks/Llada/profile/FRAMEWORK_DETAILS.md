# Setup Overview: LLaDA Model Inference & Profiling Pipeline
The LLaDA Inference & Profiling Pipeline provides a "microscopic" look into the execution dynamics of Large Language Diffusion Action models. While standard benchmarks focus on accuracy, this framework is dedicated to computational telemetry, capturing the raw interaction between the LLaDA-8B architecture and the underlying NVIDIA GPU hardware.

## Core Objectives
- GPU Workload Decomposition: The pipeline breaks down the complex "denoising" steps of the diffusion process into individual CUDA kernel executions.
- GEMM Precision Analysis: By extracting $M, N, K$ dimensions (matrix multiplication parameters), the framework identifies the specific mathematical shapes that dominate the model's workload. This is critical for optimizing matrix-multiply-accumulate (GEMM) operations that form the backbone of the transformer block.

## Profiling Dimensions
The pipeline generates an architectural "X-ray" through several technical lenses:
- Kernel Taxonomy: It categorizes GPU activity into Open-Source vs. Closed-Source kernels. This helps in understanding how much of the model's performance relies on standard libraries (like cuBLAS) versus custom-written CUDA code.
- Arithmetic Intensity (MNK Details): By logging and sorting every function call, the system identifies the "hottest" kernels. Analyzing the $M, N, K$ dimensions helps engineers determine if the model is memory-bound (limited by data movement) or compute-bound (limited by the GPU's math units).

## 5. Directory Structure & Result Aggregation
 
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

- nvjet_details.xlsx
Detailed report generated using NVIDIA profiling tools.
Contains:
Kernel-level breakdown, shapes etc

- open_closed_source_kernels.xlsx
Classification of kernels into:
Open-source kernels 
Closed-source kernels



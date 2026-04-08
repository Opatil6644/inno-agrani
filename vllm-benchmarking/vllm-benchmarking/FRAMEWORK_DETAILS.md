# vLLM Benchmarking Framework

## Overview

The **vLLM Benchmarking Framework** is a specialized diagnostic ecosystem designed to evaluate and quantify the performance characteristics of **Large Language Models (LLMs)** running on the **vLLM inference engine**.

The framework provides a standardized benchmarking environment that isolates system configuration complexity from model evaluation. This enables reliable **comparative analysis across multiple LLM architectures**.

The benchmarking suite focuses on three primary performance metrics:

- **Latency** – Time taken for the model to generate responses.
- **Throughput** – Number of tokens processed per second.
- **Serving Performance** – Performance under simulated production workloads.

In addition to standard benchmarking, the framework integrates **deep GPU profiling using NVIDIA Nsight Systems (nsys)** to capture hardware-level metrics such as:

- CUDA kernel execution
- cuBLAS GEMM operations
- Matrix multiplication dimensions (MNK)

These insights help analyze **hardware utilization, compute efficiency, and model execution behaviour**.

---

# Framework Architecture

The framework is structured into modular components to separate configuration, benchmarking execution, dataset management, profiling, and post-processing.
Each directory is responsible for a specific part of the benchmarking pipeline.

---

# Configuration Management (`config`)

The **config** directory contains all configuration files required to run benchmark experiments. These configurations are divided into three categories.

## Model Configuration

Model configuration files contain parameters required for benchmarking specific models.

Examples include:

- Tensor Parallel Size
- Trust Remote Code Flag
- Model Commit ID
- Model Repository Information

The **model commit ID** ensures that benchmarking is performed against a fixed and reproducible model version.

---

## Standard Benchmark Parameters

The framework contains configuration files defining standard parameters used by the vLLM benchmarking scripts.

These include parameters for:

- Latency tests
- Throughput tests
- Serving benchmarks

Typical parameters defined include:

- Batch sizes
- Input sequence lengths
- Output token limits
- Request concurrency levels

These ensure **consistent benchmarking conditions across models**.

---

## User Configuration

User configuration is divided into **dynamic** and **static** configurations.

### Dynamic Configuration (`dynamic_config.json`)

Contains parameters that may change between benchmark runs.

Examples:

- HuggingFace access token
- vLLM backend selection
- Compilation mode
- Attention backend
- Execution mode (eager or compiled)

These values are typically configured during runtime or in CI/CD pipelines.

---

### Static Configuration (`static_config.json`)

Contains configuration values that remain constant across experiments.

Examples:

- vLLM repository commit ID
- Framework root directory

Separating dynamic and static configurations improves **maintainability and reproducibility**.

---

# Custom vLLM Modifications (`custom-vllm-files`)

This directory contains **modified versions of standard vLLM benchmark scripts**.

Modified scripts include:

- `latency.py`
- `throughput.py`
- `serving.py`

These scripts extend the default functionality of vLLM by introducing support for additional runtime parameters such as:

- `trust_remote_code`
- compilation modes
- eager execution
- configurable attention backends
- additional runtime flags

---

# Custom Benchmark Datasets (`custom_datasets`)

The **custom_datasets** directory contains datasets used to benchmark models on real-world tasks rather than synthetic workloads.

Available datasets include:

- BBH
- GPQA
- IFEval
- MATH
- MMLU
- MUSR

Each dataset is stored as a **JSON file compatible with the vLLM benchmarking pipeline**.

Using multiple datasets enables evaluation across different **reasoning, knowledge, and problem-solving domains**.

---

# Benchmark Automation Scripts (`scripts`)

The **scripts** directory contains automation utilities used to manage benchmark configurations and model entries.

## Configuration Modification Scripts

### add_json_values.py

Purpose:

Dynamically updates vLLM benchmark JSON configurations for CI/CD pipelines.

Functionality:

- Accepts a model name from CLI arguments
- Matches the model against a CSV configuration file
- Retrieves model-specific parameters such as:
  - Tensor Parallel size
  - Trust Remote Code setting
- Updates target JSON benchmark configuration files accordingly.

This enables automated configuration generation during CI workflows.

---

### custom_test_config_modify

This module manages and filters benchmark configurations.

Key functionality:

- Processes a large master set of benchmark definitions
- Extracts a subset tailored to the current hardware environment
- Ensures compatibility with available GPUs and model architecture.

---

### test_configuration_modify.py

Purpose:

Automates the addition of new models to benchmark suites.

Functionality:

- Appends standardized benchmark configurations for:
  - Latency tests
  - Throughput tests
  - Serving tests
- Prevents duplicate entries
- Normalizes benchmark test names.

This allows the framework to easily scale to support **new models**.

---

# Benchmark Execution (`evaluate`)

The **evaluate** directory contains the primary orchestration script responsible for executing the benchmarking pipeline.

## Benchmark Orchestration

The script parses:

- `models.yaml`
- `dynamic_config.json`
- `static_config.json`

This enables batch execution across multiple LLM architectures.

---

## Dynamic Runtime Configuration

The benchmarking scripts are modified dynamically during execution to configure:

- compilation modes
- eager execution
- attention backends

This allows evaluation of multiple runtime configurations.

---

## Deep GPU Profiling

Benchmark execution is wrapped with:

This captures detailed GPU execution information including:

- CUDA kernel timelines
- cuBLAS GEMM operations
- GPU memory activity
- kernel launch statistics.

---

## Result Aggregation

After benchmark completion, the script collects:

- benchmark result JSON files
- GPU kernel profiling summaries.

These results are then passed to the post-processing pipeline.

---

# Profiling Post Processing (`post_process_profiling`)

This module processes Nsight profiling outputs and extracts actionable insights.

It contains several Python scripts responsible for analyzing GPU kernel activity.

## Kernel Analysis

Includes scripts for:

- Kernel listing
- Kernel classification
- Open-source vs closed-source kernel identification.

---

## Matrix Multiplication Analysis

The framework extracts **MNK dimensions** from GEMM kernels.

MNK refers to matrix multiplication operations:



These dimensions help analyze the compute workloads executed by the model.

---

## Aggregation Utilities

The following analysis operations are performed:

- Kernel aggregation
- MNK aggregation
- MNK sorting
- NVJET kernel aggregation
- NVJET kernel listing

These analyses help identify:

- dominant GPU kernels
- compute hotspots
- memory-bound vs compute-bound workloads.

---

# Environment Setup (`setup`)

The **setup** directory prepares the runtime environment.

## Components
- `build_setup.sh`

---

# HuggingFace Authentication (`hf_login.py`)

This script handles secure authentication with the **HuggingFace Hub**.

Functionality:

1. Loads configuration values from `config.json`
2. Extracts the HuggingFace token
3. Performs a non-interactive login to the HuggingFace Hub
4. Provides authentication status messages

This allows the framework to access:

- gated models
- private model repositories
- specific model commit versions.

---

# Model Selection (`models.yaml`)

The **models.yaml** file defines the models available for benchmarking.

Users can enable or disable models by commenting or uncommenting entries.



## File-Level Change Log 
To include custom parameter to the setup we had to do some changes to the vllm files, below are the custom parameters that we have included to the setup: 
- enforce_eager -> to run the process in either eager mode or graph mode 
- trust_remote_code -> by default it vllm sets it false, for few models this flag needs to be set true 
- vllm_compilation_mode -> to use specific vllm compilation optimization level. 


Here are the files and line numbers that we changed to make the above into effect, 
These files can be found inside custom-vllm-files directory with changes made. 

- custom-vllm-files/latency.py -> line number: 89, 123,  124 
- custom-vllm-files/throughput.py -> line number: 49, 95, 96 
- custom-vllm-files/envs.py -> line number: 218, 1408, 1409 
- custom-vllm-files/compilation.py -> line number: 180-188 

 

We also made changes to throughput.py to include the custom dataset support, here are the changes made. 
- custom-vllm-files/throughput.py -> line number:  30, 80, 389-390, 551 
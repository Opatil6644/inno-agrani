# Setup Overview: Vision Transformer (ViT)
The Vision Transformer (ViT) Automation Framework is a specialized diagnostic suite designed to bridge the gap between computer vision and high-performance transformer execution. While traditional Convolutional Neural Networks (CNNs) rely on local spatial filters, ViTs utilize global self-attention on image patches. This framework provides the deep-level telemetry required to understand how these massive "attention-over-pixels" operations translate into GPU hardware utilization.

## Core Objectives
- Architectural Efficiency Mapping: The pipeline quantifies the computational cost of the "Patch + Position" embedding process and the subsequent multi-head self-attention layers. It identifies whether the quadratic complexity of attention becomes a hardware bottleneck during image inference.
- GEMM-Level Telemetry: By extracting $M, N, K$ dimensions and cuBLAS function calls, the framework reveals the specific mathematical "footprint" of the model. This is critical for optimizing the linear projections and MLP blocks that dominate the ViT architecture.
- Hardware-Software Synergy: Utilizing deep profiling, the framework categorizes GPU activity into Open-Source vs. Closed-Source kernels, exposing the efficiency of the underlying CUDA execution paths and identifying opportunities for Tensor Core acceleration.

## Profiling Dimensions
The framework dissects the visual inference cycle through several high-resolution lenses:
- Arithmetic Intensity (MNK Analysis): It captures the exact shapes of the matrix multiplications within the Transformer Encoder. This data helps engineers determine if the model is hitting "sweet spots" for NVIDIA GPU performance based on image resolution and patch size.
- Kernel Taxonomy: It provides a granular summary of every kernel execution, distinguishing between standard library calls and custom operations. This allows for a clear understanding of the "overhead" vs. "compute" ratio in visual transformers.
- Deployment Reliability: By automating the collection of nvjet and cublas details, the framework creates a reproducible performance baseline, ensuring that architectural changes (like changing patch sizes or adding layers) are backed by empirical hardware data.
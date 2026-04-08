# Setup Overview: Wan2.1 Video Generation & Profiling
## 1. Project Overview
The Wan2.1 Video Generation Pipeline is a specialized performance-benchmarking suite centered on the Wan2.1-T2V-14B model. It automates the generation of high-fidelity video content while utilizing NVIDIA Nsight Systems to capture the underlying hardware execution timeline. This is critical for understanding the computational bottlenecks of large-scale Text-to-Video (T2V) inference.

## 2. Profiling & Hardware Analysis
The suite provides a granular look at the GPU's "Black Box" during video synthesis. All data is housed in the Profile_Details/ directory:

### Computational Efficiency
- mnk_details_sorted.csv: In the context of 14B models, this file reveals how the pipeline tiles large matrix multiplications. It is essential for identifying if the model is hitting "Tensor Core" limits.
- cublas_func_calls.csv: Tracks the specialized cuBLAS library calls used for the massive linear algebra operations required by T2V models.

### Architecture-Specific Metrics
- open_closed_source_kernels.xlsx: Essential for engineering teams to see how much of the Wan2.1 execution relies on proprietary NVIDIA libraries vs. open CUDA implementations.
- nvjet_details.xlsx: Captures thermal and power throttling events—vital for long video generation runs that can cause significant GPU heat soak.
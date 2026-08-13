# Energy-Aware GPU Kernel Colocation

This repository was used to write **Energy-Aware GPU Kernel Colocation** paper, which is also provided in it. Before using the repository it is higly recommended to read the paper, as it explains the goal of this research, the implemented strategies and other important concepts. Furthermore, it states the findings of the research project.



A CUDA-based framework for studying **GPU kernel colocation with the goal of identifying more energy efficient techniques**.

The repository contains various kernels, which originate from the Rodinia benchmark suite and CUDA SDK, namely:

- Vector addition
- Matrix multiplication
- SRAD
- Neural network 
- Hotspot
- Histogram
- Gaussian elimination
- Tensor Core GEMM
- Black-Scholes
- Binary partition
- Backpropagation
- Breadth-first search


Workloads were stripped into their minimal memory operations and kernel laucnhes. For each worworkloads a few implementations are provided, the most important being the `spatial` and `temporal` stripped variants, which are designed to be executed by the provided Framewrok. Aleternate variations are provided which can be used for error checking.

The framework provides programs to run kernels individually or concurrently and measure execution time, GPU power, and estimated energy consumption. Additioanlly, it contains files to profile kernels to measure GPU resource utilization, done with NVIDIA Nsight Compute.

> **Research / experimental software:** The code is intended primarily for GPU performance and energy experiments. The provided code was implemented for the NVIDIA GeForce RTX 3080 Ti, which has 80 Stream Multiprocessors.

## Strategies Implemented

The project investegated different strategies to identify which is best in reducing energy consumption, namely:

- **Temporal colocation** — Kernels alternate execution, sharing the same resources, but executing in parallel.
- **Spatial colocation** — Kernels execute concurrently using different resources from the GPU. Two spatial strategies were implemented, Stacked Colocation with Streams and Stacked Colocation with Green Context. 

To achieve the desired behaviours, kernel partitioning was done, where a kernel is divided into multiple subkernels.

- **Energy measurement** — uses NVIDIA Management Library (NVML) to sample GPU power and combines power with measured runtime to estimate energy.
- **Green Context experiments** — includes an experiment using CUDA Green Context / SM resource partitioning to assign portions of the GPU to concurrent workloads.


## Requirements

- An **NVIDIA GPU**
- An NVIDIA driver compatible with the CUDA Toolkit installed
- For full functionallity, the GPU should support Green Context and Nsight Compute
### Software

The main experiments require:

- **NVIDIA CUDA Toolkit**, including `nvcc`
- CUDA runtime/driver libraries
- **NVML** / NVIDIA Management Library (`libnvidia-ml`)
- A C++ compiler compatible with the CUDA Toolkit
- **C++17** support for the framework programs
# Energy-Aware GPU Kernel Colocation

This repository contains the code used for the paper **Energy-Aware GPU Kernel Colocation**, which is also included here. Before using the repository, it is highly recommended to read the paper, as it explains the research goals, the implemented strategies, key concepts, and findings.

> **Research / experimental software:** This code is intended for GPU performance and energy experiments. It was developed and tested on the **NVIDIA GeForce RTX 3080 Ti** (80 Streaming Multiprocessors). Results may differ on other hardware.

---

## Overview

A CUDA-based framework for studying GPU kernel colocation strategies with the goal of identifying more energy-efficient techniques. The framework supports running kernels individually or concurrently, and measures execution time, GPU power, and estimated energy consumption.

---

## Kernels

The following kernels are included, originating from the Rodinia benchmark suite and the CUDA Sample SDK:

- Vector addition (vecAdd)
- Matrix multiplication (mm)
- SRAD v2 (sr1, sr2)
- Neural network (nn)
- Hotspot (hot)
- Histogram (hist)
- Gaussian elimination (g1, g2)
- Tensor Core GEMM (gemm)
- Black-Scholes (bs)
- Binary partition (bip)
- Backpropagation (bp1, bp2)
- Breadth-first search (bfs1, bfs2)

Kernels were stripped to their minimal memory operations and kernel launches. For each workload, multiple implementations are provided. The most important are the `spatial` and `temporal` stripped variants, designed to be used with the provided framework. Alternate variations are included for error checking and validation purposes.

---

## Colocation Strategies

Three strategies were implemented and evaluated:

- **Sequential execution (baseline)** — kernels run one after another on the same stream, with no concurrency.
- **Stacked colocation with Streams** — kernels execute concurrently on separate CUDA streams. Kernel partitioning is used to limit blocks per subkernel to half the number of available SMs, enabling spatial sharing.
- **Stacked colocation with Green Context** — kernels execute concurrently with hardware-enforced SM partitioning via CUDA Green Context, guaranteeing resource isolation between kernels.
- **Temporal colocation** — kernels are divided into subkernels that take turns occupying the full GPU, so interleaved execution.

To achieve the desired execution behaviour, **kernel partitioning** was implemented, which is the process of dividng a kernel into multiple subkernels. (read paper for clarification)
---

## Requirements

### Hardware
- An NVIDIA GPU 
- Green Context experiments additionally require a GPU that support CUDA Green Context 

### Software
- **NVIDIA CUDA Toolkit** including `nvcc`
- **NVML** — NVIDIA Management Library (`libnvidia-ml`), used for power and energy measurement
- A C++ compiler compatible with the CUDA Toolkit
- **C++17** support

---

## Citation

If you use this code or build on this work, please cite:

```
Pedro Cecchini Said. Energy-Aware GPU Kernel Colocation.
TScIT 45, July 3, 2026, Enschede, The Netherlands. University of Twente.
```

---
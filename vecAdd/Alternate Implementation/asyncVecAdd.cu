/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * Sweeps a set of stream-partition counts {1,2,4,8,16}, runs several
 * timed trials at each partition count, and reports averaged
 * elapsed-time / power / energy numbers per partition count.
 */

 // run command nvcc asyncVecAdd.cu -o asyncvecadd
// ./asyncvecadd
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <nvml.h>

// Partition counts to sweep, and how many timed trials to average per count.
static const int PARTITION_SIZES[] = {1,2,3,4,6,8,16};
static const int NUM_CONFIGS = sizeof(PARTITION_SIZES) / sizeof(PARTITION_SIZES[0]);
#define NUM_TRIALS 10

nvmlDevice_t device;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    #pragma unroll 1
    for (int j = 0; j < 10200000; j++) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < numElements) {
            C[i] = A[i] + B[i] + 0.0f;
        }
    }
}

/**
 * Runs NUM_TRIALS timed trials of the partitioned vector-add for a given
 * number of partitions, verifying correctness each trial and accumulating
 * elapsed time / average power / energy. Writes the per-trial averages
 * to the output pointers.
 */
void runPartitionConfig(int numPartitions,
                         float *d_A, float *d_B, float *d_C,
                         float *h_A, float *h_B, float *h_C,
                         int numElements, size_t size,
                         float *outAvgTimeMs, float *outAvgPowerW, float *outAvgEnergyJ)
{
    int chunkSize = numElements / numPartitions;
    int threadsPerBlock = 256;
    int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t *streams = (cudaStream_t *)malloc(numPartitions * sizeof(cudaStream_t));
    for (int i = 0; i < numPartitions; i++) {
        cudaStreamCreate(&streams[i]);
    }

    float sumTimeMs = 0.0f;
    float sumPowerW = 0.0f;
    float sumEnergyJ = 0.0f;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        cudaEvent_t globalStart, globalStop;
        cudaEventCreate(&globalStart);
        cudaEventCreate(&globalStop);

        unsigned int powerBefore, powerAfter;
        nvmlDeviceGetPowerUsage(device, &powerBefore);

        cudaEventRecord(globalStart);
        for (int i = 0; i < numPartitions; i++) {
            int offset = i * chunkSize;
            vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
                d_A + offset, d_B + offset, d_C + offset, chunkSize);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(globalStop);
        cudaEventSynchronize(globalStop);

        nvmlDeviceGetPowerUsage(device, &powerAfter);

        float msTotal;
        cudaEventElapsedTime(&msTotal, globalStart, globalStop);
        float avgPowerW = ((powerBefore + powerAfter) / 2.0f) / 1000.0f;
        float energyJ = avgPowerW * (msTotal / 1000.0f);

        //   printf("partitions: %3d | avg time: %10.3f ms | avg power: %10.3f W | avg energy: %10.3f J\n",
        //        numPartitions, msTotal, avgPowerW, energyJ);

        cudaEventDestroy(globalStart);
        cudaEventDestroy(globalStop);

        sumTimeMs += msTotal;
        sumPowerW += avgPowerW;
        sumEnergyJ += energyJ;

        // Verify correctness for this trial.
        cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        int coveredElements = chunkSize * numPartitions;
        for (int i = 0; i < coveredElements; ++i) {
            if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
                fprintf(stderr,
                        "Result verification failed (partitions=%d, trial=%d, element=%d)!\n",
                        numPartitions, trial, i);
                exit(EXIT_FAILURE);
            }
        }
    }

    for (int i = 0; i < numPartitions; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);

    *outAvgTimeMs = sumTimeMs / NUM_TRIALS;
    *outAvgPowerW = sumPowerW / NUM_TRIALS;
    *outAvgEnergyJ = sumEnergyJ / NUM_TRIALS;
}

/**
 * Host main routine
 */
int main(void)
{
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);

    int numElements = 240 * 256;
    size_t size = numElements;
    printf("[Vector addition of %d elements]\n", numElements);

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, size * sizeof(float));
    cudaMallocHost(&h_B, size * sizeof(float));
    cudaMallocHost(&h_C, size * sizeof(float));

    // Initialize the host input vectors (fixed for all partition configs).
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    printf("Sweeping partition counts: ");
    for (int c = 0; c < NUM_CONFIGS; c++) {
        printf("%d%s", PARTITION_SIZES[c], (c == NUM_CONFIGS - 1) ? "\n" : ", ");
    }
    printf("Averaging over %d trials per partition count.\n\n", NUM_TRIALS);

    for (int c = 0; c < NUM_CONFIGS; c++) {
        int numPartitions = PARTITION_SIZES[c];
        float avgTimeMs, avgPowerW, avgEnergyJ;

        runPartitionConfig(numPartitions, d_A, d_B, d_C, h_A, h_B, h_C,
                            numElements, size, &avgTimeMs, &avgPowerW, &avgEnergyJ);

        printf("partitions: %3d | avg time: %10.3f ms | avg power: %10.3f W | avg energy: %10.3f J\n",
               numPartitions, avgTimeMs, avgPowerW, avgEnergyJ);
    }

    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    nvmlShutdown();

    printf("\nDone\n");
    return 0;
}

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

/*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */
// To run this code use the framework
// 
// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h> // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"


namespace histogram{
static uchar              *h_Data;
static  uint               *h_HistogramCPU, *h_HistogramGPU;
static  uchar              *d_Data;
static  uint               *d_Histogram;
static  int                 PassFailFlag = 1;
static  uint                byteCount    = 64 * 1048576;
static  uint                uiSizeMult   = 1;
static int          numRuns    = 16;
static const uint PARTIAL_HISTOGRAM256_COUNT = 320; //  Kept thread size as original, but consequently needs to send more blocks Total thread = 61440
                                                //for 65536 threads
 static int launches;

static int chunkBytes, partialPerPartition;


int setup(int partitions, cudaStream_t stream){
    

    cudaMallocHost(&h_Data, byteCount);
    cudaMallocHost(&h_HistogramCPU, HISTOGRAM256_BIN_COUNT * sizeof(uint));
    cudaMallocHost(&h_HistogramGPU, HISTOGRAM256_BIN_COUNT * sizeof(uint));


    

    // printf("...generating input data\n");
    srand(2009);

    for (uint i = 0; i < byteCount; i++) {
        h_Data[i] = rand() % 256;
    }

    // printf("Initializing 256-bin histogram...\n");
    initHistogram256();

    // printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);

    int blocksPerLaunch = 40;
    launches            = PARTIAL_HISTOGRAM256_COUNT / blocksPerLaunch;  
    chunkBytes          = byteCount / launches;          
    partialPerPartition = blocksPerLaunch;               

    printf("Launched %d kernels with: %d blocks, with Threads: %d\n", launches, partialPerPartition, HISTOGRAM256_THREADBLOCK_SIZE);


    checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, (HISTOGRAM256_BIN_COUNT) * sizeof(uint)));


    checkCudaErrors(cudaMemcpyAsync(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice, stream));

    return 0;
}

int launcher(int partitions, cudaStream_t stream){
    for(int i = 0; i < launches; i++){
            int dataOffset = i * (chunkBytes/sizeof(uint));
            int partialOffset = i * partialPerPartition;

            strippedhistogram256(d_Histogram,
                        (uint*)d_Data + dataOffset,
                        chunkBytes,
                        stream,
                        partialOffset,
                        partialPerPartition);
        }

    return 0;
}

int cleaner(cudaStream_t stream){
    printf("histogram cleaner freeing: d_histogram=%p d_Data=%p h_HistogramGPU=%p h_Data=%p\n",
       d_Histogram, d_Data, h_HistogramGPU, h_Data);
    cudaMemcpyAsync(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();
    closeHistogram256();       

    checkCudaErrors(cudaFree(d_Histogram));
    checkCudaErrors(cudaFree(d_Data));
    cudaFreeHost(h_HistogramGPU);
    cudaFreeHost(h_HistogramCPU);
    cudaFreeHost(h_Data);
    return 0;
}


}
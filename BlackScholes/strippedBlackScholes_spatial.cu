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
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

 //This code should be ran using the framework

#include <helper_cuda.h>      // helper functions CUDA error checking and initialization
#include <helper_functions.h> // helper functions for string parsing
#include "BlackScholes_kernel.cuh"


namespace blackScholes{
      
    static float
        // Results calculated by CPU for reference
        *h_CallResultCPU,
        *h_PutResultCPU,
        // CPU copy of GPU results
        *h_CallResultGPU, *h_PutResultGPU,
        // CPU instance of input data
        *h_StockPrice, *h_OptionStrike, *h_OptionYears;

    static float
        // Results calculated by GPU
        *d_CallResult,
        *d_PutResult,
        // GPU instance of input data
        *d_StockPrice, *d_OptionStrike, *d_OptionYears;
    static int chunkSize;
    static dim3 dimBlock, dimGrid;
    static int launches;


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult,
                                float *h_PutResult,
                                float *h_StockPrice,
                                float *h_OptionStrike,
                                float *h_OptionYears,
                                float  Riskfree,
                                float  Volatility,
                                int    optN);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N          = 131072;

const int   OPT_SZ     = OPT_N * sizeof(float);
const float RISKFREE   = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))


int setup(int partitions, cudaStream_t stream){

    
    cudaMallocHost(&h_CallResultCPU, OPT_SZ);
    cudaMallocHost(&h_PutResultCPU, OPT_SZ);
    cudaMallocHost(&h_StockPrice, OPT_SZ);
    cudaMallocHost(&h_OptionStrike, OPT_SZ);
    cudaMallocHost(&h_OptionYears, OPT_SZ);
    cudaMallocHost(&h_CallResultGPU, OPT_SZ);
    cudaMallocHost(&h_PutResultGPU, OPT_SZ);

    checkCudaErrors(cudaMalloc((void **)&d_CallResult, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_PutResult, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_OptionYears, OPT_SZ));

    srand(5347);

    // Generate options set
    for (int i = 0; i < OPT_N; i++) {
        h_CallResultCPU[i] = 0.0f;
        h_PutResultCPU[i]  = -1.0f;
        h_StockPrice[i]    = RandFloat(5.0f, 30.0f);
        h_OptionStrike[i]  = RandFloat(1.0f, 100.0f);
        h_OptionYears[i]   = RandFloat(0.25f, 10.0f);
    }

    int blocksPerLaunch = 40;
    int totalBlocks = DIV_UP(OPT_N/2, 256);
    launches = totalBlocks/blocksPerLaunch;
    dimBlock = dim3(256);
    dimGrid  = dim3(blocksPerLaunch);

    chunkSize = totalBlocks/launches;

    printf("Lauched %d kernels, of %d blocks, with %d threads\n", launches,dimGrid.x, dimBlock.x);

    checkCudaErrors(cudaMemcpyAsync(d_StockPrice, h_StockPrice , OPT_N * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_OptionStrike , h_OptionStrike , OPT_N * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_OptionYears , h_OptionYears, OPT_N * sizeof(float), cudaMemcpyHostToDevice, stream));

    return 0;
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int launcher(int partitions, cudaStream_t stream)
{
    for (int i = 0; i < launches; i++){   
        int offset = i*chunkSize;
       


        BlackScholesGPU<<<dimGrid, dimBlock,0, stream /*512, 128*/>>>((float2 *)d_CallResult +offset/2,
                                                                            (float2 *)d_PutResult+offset/2,
                                                                            (float2 *)d_StockPrice+offset/2,
                                                                            (float2 *)d_OptionStrike+offset/2,
                                                                            (float2 *)d_OptionYears+offset/2,
                                                                            RISKFREE,
                                                                            VOLATILITY,
                                                                            offset,
                                                                            OPT_N);
            getLastCudaError("BlackScholesGPU() execution failed\n");
        
    
    }

    return 0;
}

int cleaner(cudaStream_t stream){

        checkCudaErrors(cudaMemcpyAsync(h_CallResultGPU, d_CallResult, OPT_N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        checkCudaErrors(cudaMemcpyAsync(h_PutResultGPU, d_PutResult, OPT_N * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaDeviceSynchronize();


        checkCudaErrors(cudaFree(d_OptionYears));
        checkCudaErrors(cudaFree(d_OptionStrike));
        checkCudaErrors(cudaFree(d_StockPrice));
        checkCudaErrors(cudaFree(d_PutResult));
        checkCudaErrors(cudaFree(d_CallResult));

        cudaFreeHost(h_OptionYears);
        cudaFreeHost(h_OptionStrike);
        cudaFreeHost(h_StockPrice);
        cudaFreeHost(h_PutResultGPU);
        cudaFreeHost(h_CallResultGPU);
        cudaFreeHost(h_PutResultCPU);
        cudaFreeHost(h_CallResultCPU);

    return 0;
    }
}
//Framework

#include "../vecAdd/strippedVecAdd_spatial.cu"
#include "../matrixMultiplication/strippedMatrixMultiplication_spatial.cu"
#include "../srad_v2/strippedsradKernel1_spatial.cu"
#include "../srad_v2/strippedsradKernel2_spatial.cu"
#include "../nn/strippednn_spatial.cu"
#include "../hotspot/strippedhotspot_spatial.cu"
#include "../histogram/strippedHistogram_spatial.cpp"
#include "../histogram/histogramKernel.cu"
#include "../gaussian/strippedgaussian1_spatial.cu"
#include "../gaussian/strippedgaussian2_spatial.cu"
#include "../cudaTensorCoreGemm/strippedTensorCoreGemm_spatial.cu"
#include "../BlackScholes/strippedBlackScholes_spatial.cu"
#include "../binaryPartition/strippedBinaryPartition_spatial.cu"
#include "../backprop/strippedbackprop1_spatial.cu"
#include "../backprop/strippedbackprop2_spatial.cu"
#include "../bfs/strippedbfs1_spatial.cu"
#include "../bfs/strippedbfs2_spatial.cu"

#include <nvml.h>


// This file measures the performance of all kernel pairs under stacked colocation.
// Each kernel uses half of the GPU resources, achieved through manipulating launch configurations.
// The average time, power and energy is calculated for the two kernels over several iterations.


// HOW TO COMPILE: nvcc -std=c++17 StackedColocation.cu -I../Common -lnvidia-ml -o StackedColocation
//  CUDA_MODULE_LOADING=EAGER ./StackedColocation



#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#define COLOC_SIZE 2

typedef struct {
    const char *kernelA;
    const char *kernelB;
    int partitions;
    float avgTime;
    float avgPower;
    float avgEnergy;
    float avgTailTime;  
} PairResult;

typedef int (*SetupFn)(int partitions, cudaStream_t stream);
typedef int (*LauncherFn)(int partitions, cudaStream_t stream);
typedef int (*CleanerFn)( cudaStream_t stream);


typedef struct {
    const char *name;
    SetupFn setup;
    LauncherFn launcher;
    CleanerFn cleaner;
} KernelEntry;



KernelEntry kernel_registry[] = {
    { "vecAdd", vecAdd::setup, vecAdd::launcher, vecAdd::cleaner},
    { "matrixMultiplication", mm::setup, mm::launcher, mm::cleaner},
    { "sradv2_1", sradv2_1::setup, sradv2_1::launcher, sradv2_1::cleaner},
    { "sradv2_2", sradv2_2::setup, sradv2_2::launcher, sradv2_2::cleaner},
    { "nn", nn::setup, nn::launcher, nn::cleaner},
    {"hotspot", hotspot::setup, hotspot::launcher, hotspot::cleaner},
    {"histogram", histogram::setup, histogram::launcher, histogram::cleaner},
    {"gaussian1", gaussian1::setup, gaussian1::launcher, gaussian1::cleaner},
    {"gaussian2", gaussian2::setup, gaussian2::launcher, gaussian2::cleaner},
    {"gemm", gemm::setup, gemm::launcher, gemm::cleaner},
    { "blackScholes", blackScholes::setup, blackScholes::launcher, blackScholes::cleaner},
    { "binaryPartition", binaryPartition::setup, binaryPartition::launcher, binaryPartition::cleaner},
    { "backprop1", backprop1::setup, backprop1::launcher, backprop1::cleaner},
    { "backprop2", backprop2::setup, backprop2::launcher, backprop2::cleaner},
    { "bfs1", bfs1::setup, bfs1::launcher, bfs1::cleaner},
    { "bfs2", bfs2::setup, bfs2::launcher, bfs2::cleaner},
};
nvmlDevice_t device;

int num_kernels = sizeof(kernel_registry) / sizeof(kernel_registry[0]);
float averageTimeA = 0.0f;
float averageTimeB = 0.0f;
float averageTotalTime = 0.0f;


LauncherFn find_launcher(const char *name)
{
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].launcher;

    fprintf(stderr, "ERROR: kernel  launcher'%s' not found\n", name);
    return NULL;
}

CleanerFn find_cleaner(const char *name)
{
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].cleaner;

    fprintf(stderr, "ERROR: kernel cleaner'%s' not found\n", name);
    return NULL;
}

SetupFn find_setup(const char *name)
{
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].setup;

    fprintf(stderr, "ERROR: kernel setup'%s' not found\n", name);
    return NULL;
}



void warmup(int iterations) {
    const char *warmupKernel= "matrixMultiplication";
    SetupFn warmupSetup = find_setup(warmupKernel);
    LauncherFn warmupLauncher = find_launcher(warmupKernel);
    CleanerFn warmupCleaner = find_cleaner(warmupKernel);
    
    

    for (int i = 0; i < iterations; i++) {
        printf("\nWarmup run %d:\n", i+1);
        
        warmupSetup(1, 0);
        cudaDeviceSynchronize();

        warmupLauncher(1, 0);
        warmupCleaner(0);
    }


}


void colocateKernels(const char *kernelA, const char *kernelB, int partitions_A, int partitions_B, 
                     cudaStream_t streamA, cudaStream_t streamB,
                     float *energy, float *power, float *time, float *tailTimeAcc)  // pointers
{
    SetupFn    setupA    = find_setup(kernelA);
    SetupFn    setupB    = find_setup(kernelB);
    LauncherFn launcherA = find_launcher(kernelA);
    LauncherFn launcherB = find_launcher(kernelB);
    CleanerFn  cleanerA  = find_cleaner(kernelA);
    CleanerFn  cleanerB  = find_cleaner(kernelB);

    if (!launcherA || !launcherB || !cleanerA || !cleanerB || !setupA || !setupB) {
        printf("Error finding kernel functions\n");
        exit(0);
    }

    setupA(partitions_A, streamA);
    setupB(partitions_B, streamB);

    cudaEvent_t globalStart, globalStop, startA, stopA, startB, stopB;
    cudaEventCreate(&globalStart); cudaEventCreate(&globalStop);
    cudaEventCreate(&startA);      cudaEventCreate(&stopA);
    cudaEventCreate(&startB);      cudaEventCreate(&stopB);

    cudaStream_t syncStream;
    cudaStreamCreateWithFlags(&syncStream, cudaStreamNonBlocking);

    cudaDeviceSynchronize();
    unsigned int powerBefore, powerAfter;
    nvmlDeviceGetPowerUsage(device, &powerBefore);

    cudaEventRecord(globalStart, streamA);
    cudaEventRecord(startA, streamA);
    launcherA(partitions_A, streamA);
    cudaEventRecord(stopA, streamA);

    cudaEventRecord(startB, streamB);
    launcherB(partitions_B, streamB);
    cudaEventRecord(stopB, streamB);

    cudaStreamWaitEvent(syncStream, stopA, 0);
    cudaStreamWaitEvent(syncStream, stopB, 0);
    cudaEventRecord(globalStop, syncStream);

    cudaDeviceSynchronize();
    nvmlDeviceGetPowerUsage(device, &powerAfter);

    float msTotal;
    float msA, msB;
    cudaEventElapsedTime(&msA, startA, stopA);
    cudaEventElapsedTime(&msB, startB, stopB);

    float tailTime = fabs(msA - msB);


    *tailTimeAcc += tailTime;


    cudaEventElapsedTime(&msTotal, globalStart, globalStop);

    float avgPowerW = ((powerBefore + powerAfter) / 2.0f) / 1000.0f;


    *time += msTotal;
    *power     += avgPowerW;
    *energy    += avgPowerW * (msTotal / 1000.0f);

    cleanerA(streamA);
    cleanerB(streamB);

    cudaEventDestroy(startA);   cudaEventDestroy(stopA);
    cudaEventDestroy(startB);   cudaEventDestroy(stopB);
    cudaEventDestroy(globalStart); cudaEventDestroy(globalStop);
    cudaStreamDestroy(syncStream);

}

int main(void)
{
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);

    int iterations = 5; 

    // Do not change this value
    int baselinePartition = 1;

    // Change kernels to be measured.
    const char *all_kernels[] = {"sradv2_1","sradv2_2", "histogram","backprop1","hotspot", "matrixMultiplication","gaussian2", "blackScholes","gaussian1","backprop2", "bfs1", "bfs2","vecAdd","nn","binaryPartition","gemm"}; 
    int numKernels = sizeof(all_kernels) / sizeof(all_kernels[0]);

    int numPairs       = (numKernels * (numKernels - 1) / 2);

    // allocate results array
    PairResult *results = (PairResult*)malloc(numPairs * sizeof(PairResult));
    int resultIdx = 0;

    cudaStream_t streams[COLOC_SIZE];
        for (int i = 0; i < COLOC_SIZE; i++)
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaStream_t streamA = streams[0];
    cudaStream_t streamB = streams[1];
    warmup(50);

    for (int i = 0; i < numKernels; i++) {
        cudaDeviceSynchronize();                    
        cudaError_t err = cudaGetLastError();   
        if (err != cudaSuccess)
            printf("*** Sticky error at start of i=%d (%s): %s\n", 
               i, all_kernels[i], cudaGetErrorString(err));
        const char *kernelA = all_kernels[i];
        for (int j = i + 1; j < numKernels; j++) {
            const char *kernelB = all_kernels[j];
            
            int partitions_A = baselinePartition;
            int partitions_B = baselinePartition;

            float energyAcc = 0.0f;
            float powerAcc  = 0.0f;
            float timeAcc   = 0.0f;
            float tailTimeAcc = 0.0f;  

            printf("------------------Colocating kernels %s and %s-----------------------\n", kernelA, kernelB);
            for (int m = 0; m < iterations; m++) {
                colocateKernels(kernelA, kernelB, partitions_A, partitions_B, streamA, streamB, &energyAcc, &powerAcc, &timeAcc, &tailTimeAcc);
            }

          results[resultIdx++] = {
                kernelA,
                kernelB,
                partitions_A,
                timeAcc     / iterations,
                powerAcc    / iterations,
                energyAcc   / iterations,
                tailTimeAcc / iterations
            };
        
        }
    }


    // print everything at the end
    printf("\n%-20s %-20s %4s | %12s %12s %12s %12s\n",
        "KernelA", "KernelB", "P",
        "AvgTime(ms)", "AvgPower(W)",
        "AvgEnergy(J)", "AvgTail(ms)");
    printf("%-20s %-20s %4s | %12s %12s %12s %12s\n",
        "--------------------",
        "--------------------",
        "----",
        "------------",
        "------------",
        "------------",
        "------------");
    for (int i = 0; i < resultIdx; i++) {
        printf("%-20s %-20s %4d | %12.3f %12.2f %12.4f %12.3f\n",
       results[i].kernelA,
       results[i].kernelB,
       results[i].partitions,
       results[i].avgTime,
       results[i].avgPower,
       results[i].avgEnergy,
       results[i].avgTailTime);
    }

    free(results);
    for (int i = 0; i < COLOC_SIZE; i++) {
        cudaStreamDestroy(streams[i]);
    }

    nvmlShutdown();
    return 0;
}
//Framework

#include "../vecAdd/strippedVecAdd_temporal.cu"
#include "../matrixMultiplication/strippedMatrixMultiplication_temporal.cu"
#include "../srad_v2/strippedsradKernel1_temporal.cu"
#include "../srad_v2/strippedsradKernel2_temporal.cu"
#include "../nn/strippednn_temporal.cu"
#include "../hotspot/strippedhotspot_temporal.cu"
#include "../histogram/strippedHistogram_temporal.cpp"
#include "../histogram/histogramKernel.cu"
#include "../gaussian/strippedgaussian1_temporal.cu"
#include "../gaussian/strippedgaussian2_temporal.cu"
#include "../cudaTensorCoreGemm/strippedTensorCoreGemm_temporal.cu"
#include "../BlackScholes/strippedBlackScholes_temporal.cu"
#include "../binaryPartition/strippedBinaryPartition_temporal.cu"
#include "../backprop/strippedbackprop1_temporal.cu"
#include "../backprop/strippedbackprop2_temporal.cu"
#include "../bfs/strippedbfs1_temporal.cu"
#include "../bfs/strippedbfs2_temporal.cu"

#include <nvml.h>

// This file is used to get a baseline comparision for colocation. 
// Two kernels are executed stricttly sequential, with no overlap. The average time, power and energy is calculated for the two kernels over several iterations.


// HOW TO COMPILE: nvcc -std=c++17 BaselineCalculation.cu -I../Common -lnvidia-ml -o baselineColocation
//  CUDA_MODULE_LOADING=EAGER ./baselineColocation


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
                     cudaStream_t streamAB,  
                     float *energy, float *power, float *time)  // pointers
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

    setupA(partitions_A, streamAB);
    setupB(partitions_B, streamAB);


    cudaEvent_t globalStart, globalStop;
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);
 
    cudaDeviceSynchronize();
    unsigned int powerBefore, powerAfter;
    nvmlDeviceGetPowerUsage(device, &powerBefore);

    cudaEventRecord(globalStart, streamAB);
    launcherA(partitions_A, streamAB);
    cudaDeviceSynchronize();
    launcherB(partitions_B, streamAB);
    cudaEventRecord(globalStop, streamAB);

    cudaDeviceSynchronize();
    nvmlDeviceGetPowerUsage(device, &powerAfter);

    float msTotal;
    cudaEventElapsedTime(&msTotal, globalStart, globalStop);

    float avgPowerW = ((powerBefore + powerAfter) / 2.0f) / 1000.0f;

    *power  += avgPowerW;
    *energy += avgPowerW * (msTotal / 1000.0f);
    *time   += msTotal;

    cleanerA(streamAB);
    cleanerB(streamAB);
    cudaDeviceSynchronize();

    cudaEventDestroy(globalStart);
    cudaEventDestroy(globalStop);

}

int main(void)
{
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);

    int iterations = 10; 
    int baselinePartition = 1;

    //Modify kernel list here
    // const char *all_kernels[] = {"gaussian1","vecAdd", "matrixMultiplication", "sradv2_1", "sradv2_2", "nn", "hotspot", "histogram", "gaussian2", "gemm", "blackScholes", "binaryPartition", "backprop1", "backprop2", "bfs1", "bfs2"};
    const char *all_kernels[] = {"gaussian1","gemm"};

    int numKernels = sizeof(all_kernels) / sizeof(all_kernels[0]);

    int numPairs       = (numKernels *2) -1;

    // allocate results array
    PairResult *results = (PairResult*)malloc(numPairs * sizeof(PairResult));
    int resultIdx = 0;

    cudaStream_t streamAB;
    cudaStreamCreateWithFlags(&streamAB, cudaStreamNonBlocking);
    warmup(50);

    for (int i = 0; i < 1; i++) {
        const char *kernelA = all_kernels[i];
        for (int j = i + 1; j < numKernels; j++) {
            const char *kernelB = all_kernels[j];
            
            int partitions_A = baselinePartition;
            int partitions_B = baselinePartition;

            float energyAcc = 0.0f;
            float powerAcc  = 0.0f;
            float timeAcc   = 0.0f;
            printf("------------------Warmup for kernels %s and %s-----------------------\n", kernelA, kernelB);

            printf("------------------Colocating kernels %s and %s-----------------------\n", kernelA, kernelB);
            for (int m = 0; m < iterations; m++) {
                colocateKernels(kernelA, kernelB, partitions_A, partitions_B, streamAB,&energyAcc, &powerAcc, &timeAcc);
            }

            results[resultIdx++] = {
            kernelA, kernelB, partitions_A,
            timeAcc   / iterations,
            powerAcc  / iterations,
            energyAcc / iterations
            };
        
        }
    }


    // print everything at the end
    printf("\n%-20s %-20s %4s | %12s %12s %12s\n",
           "KernelA", "KernelB", "P", "AvgTime(ms)", "AvgPower(W)", "AvgEnergy(J)");
    printf("%-20s %-20s %4s | %12s %12s %12s\n",
           "--------------------", "--------------------", "----",
           "------------", "------------", "------------");

    for (int i = 0; i < resultIdx; i++) {
        printf("%-20s %-20s %4d | %12.3f %12.2f %12.4f\n",
               results[i].kernelA, results[i].kernelB, results[i].partitions,
               results[i].avgTime, results[i].avgPower, results[i].avgEnergy);
    }

    free(results);
    cudaStreamDestroy(streamAB);

    nvmlShutdown();
    return 0;
}
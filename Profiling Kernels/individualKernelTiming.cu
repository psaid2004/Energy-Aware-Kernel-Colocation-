//Framework

#include "../vecAdd/strippedVecAdd_256.cu"
#include "../matrixMultiplication/strippedMatrixMultiplication_256.cu"
#include "../srad_v2/strippedsradKernel1_256.cu"
#include "../srad_v2/strippedsradKernel2_256.cu"
#include "../nn/strippednn_256.cu"
#include "../hotspot/strippedhotspot_256.cu"
#include "../histogram/strippedHistogram_256.cpp"
#include "../histogram/histogramKernel.cu"
#include "../gaussian/strippedgaussian1_256.cu"
#include "../gaussian/strippedgaussian2_256.cu"
#include "../cudaTensorCoreGemm/strippedTensorCoreGemm_256.cu"
#include "../BlackScholes/strippedBlackScholes_256.cu"
#include "../binaryPartition/strippedBinaryPartition_256.cu"
#include "../backprop/strippedbackprop1_256.cu"
#include "../backprop/strippedbackprop2_256.cu"
#include "../bfs/strippedbfs1_256.cu"
#include "../bfs/strippedbfs2_256.cu"

#include <nvml.h>




// HOW TO COMPILE: nvcc -std=c++17 BaselineCalculation.cu -I../Common -lnvidia-ml -o baselinetest


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#define COLOC_SIZE 2


typedef int (*SetupFn)(int partitions, cudaStream_t stream);
typedef int (*LauncherFn)(int partitions, cudaStream_t stream);
typedef int (*CleanerFn)( cudaStream_t stream);

typedef struct {
    const char *kernelA;
    const char *kernelB;
    int partitions;
    float avgTime;
    float avgPower;
    float avgEnergy;
} PairResult;
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
    { "gaussian1",           (SetupFn)gaussian1::setup,       (LauncherFn)gaussian1::launcher,       (CleanerFn)gaussian1::cleaner       },
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



void warmup() {
    const char *warmupKernel= "matrixMultiplication";
    SetupFn warmupSetup = find_setup(warmupKernel);
    LauncherFn warmupLauncher = find_launcher(warmupKernel);
    CleanerFn warmupCleaner = find_cleaner(warmupKernel);
    
    

    for (int i = 0; i < 50; i++) {
        printf("\nWarmup run %d:\n", i+1);
        
        warmupSetup(1, 0);
        cudaDeviceSynchronize();

        warmupLauncher(1, 0);
        warmupCleaner(0);
    }

}


void colocateKernels(const char *kernelA, const char *kernelB, int partitions_A, int partitions_B, cudaStream_t streamA, cudaStream_t streamB, float *energy, float *power, float *time){
    SetupFn    setupA    = find_setup(kernelA);
    LauncherFn launcherA = find_launcher(kernelA);
    CleanerFn  cleanerA  = find_cleaner(kernelA);

    cudaDeviceSynchronize();
    if (!launcherA || !cleanerA || !setupA) {
        printf("Error finding kernel functions\n");
        exit(0);
    }

    setupA(partitions_A, streamA);


    cudaEvent_t globalStart, globalStop;
    cudaEventCreate(&globalStart);
    cudaEventCreate(&globalStop);
 
    cudaDeviceSynchronize();
    unsigned int powerBefore, powerAfter;
    nvmlDeviceGetPowerUsage(device, &powerBefore);

    cudaEventRecord(globalStart, streamA);
    launcherA(partitions_A, streamA);
    cudaEventRecord(globalStop, streamA);

    cudaDeviceSynchronize();
    nvmlDeviceGetPowerUsage(device, &powerAfter);

    float msTotal;
    cudaEventElapsedTime(&msTotal, globalStart, globalStop);

    float avgPowerW = ((powerBefore + powerAfter) / 2.0f) / 1000.0f;

    *power  += avgPowerW;
    *energy += avgPowerW * (msTotal / 1000.0f);
    *time   += msTotal;

    cleanerA(streamA);
    cudaDeviceSynchronize();

    cudaEventDestroy(globalStart);
    cudaEventDestroy(globalStop);
   
   
}


int main(void){
   

    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);
    int iterations = 1;
    const char *all_kernels[] =  {"vecAdd", "matrixMultiplication", "sradv2_1", "sradv2_2", "nn", "hotspot", "histogram", "gaussian1", "gaussian2", "gemm", "blackScholes", "binaryPartition", "backprop1", "backprop2", "bfs1", "bfs2"};
    // const char *all_kernels[] =  {"vecAdd", "matrixMultiplication"};

    cudaStream_t streams[COLOC_SIZE];    
    // Initialize streams
    for (int i = 0; i < COLOC_SIZE; i++){
                cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }
    cudaStream_t streamA = streams[0];
    cudaStream_t streamB = streams[0];
    
    int numKernels = sizeof(all_kernels) / sizeof(all_kernels[0]);
    int numPairs       = (numKernels * (numKernels - 1) / 2);
    PairResult *results = (PairResult*)malloc(numPairs * sizeof(PairResult));
    int resultIdx = 0;
    const char *kernelA = all_kernels[0];

    warmup();


   
   for (int i = 0; i < sizeof(all_kernels) / sizeof(all_kernels[0]); i++){
        kernelA = all_kernels[i];
        int partitions_A = 1;
        printf("Kernel1 %s, Num of partitions %d  \n", kernelA, partitions_A);
        // printf("Finished Warmup\n");
        float energyAcc = 0.0f;
        float powerAcc  = 0.0f;
        float timeAcc   = 0.0f;
        for (int m = 0; m < iterations; m++){
                colocateKernels(kernelA, kernelA, partitions_A, partitions_A, streamA, streamB, &energyAcc, &powerAcc, &timeAcc);
               }
        results[resultIdx++] = {
            kernelA, kernelA, partitions_A,
            timeAcc   / iterations,
            powerAcc  / iterations,
            energyAcc / iterations
            };
    }


    for (int i = 0; i < COLOC_SIZE; i++){
        cudaStreamDestroy(streams[i]);
    }

printf("\n%-20s %4s | %12s %12s %12s\n",
       "Kernel", "P", "AvgTime(ms)", "AvgPower(W)", "AvgEnergy(J)");
printf("%-20s %4s | %12s %12s %12s\n",
       "--------------------", "----",
       "------------", "------------", "------------");
for (int i = 0; i < resultIdx; i++)
    printf("%-20s %4d | %12.3f %12.2f %12.4f\n",
           results[i].kernelA, results[i].partitions,
           results[i].avgTime, results[i].avgPower, results[i].avgEnergy);

free(results);


    nvmlShutdown();

    return 0;
    

    }
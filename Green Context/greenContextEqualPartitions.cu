//Framework
#define CUDA_DRIVER_API         
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

// nvcc -std=c++17 greenContextEqualPartitions.cu -I../Common -lcuda -lnvidia-ml -o greenContextTest
// CUDA_MODULE_LOADING=EAGER nsys profile --trace=cuda,nvtx,osrt --gpu-metrics-devices=0 --force-overwrite=true -o nsysGreenContext ./greenContextTest

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef int (*SetupFn)(int partitions, cudaStream_t stream);
typedef int (*LauncherFn)(int partitions, cudaStream_t stream);
typedef int (*CleanerFn)(cudaStream_t stream);


typedef struct {
    const char *kernelA;
    const char *kernelB;
    int partitions;
    int smsA;
    int smsB;
    float avgTimeTotal;
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
    { "vecAdd",               vecAdd::setup,           vecAdd::launcher,          vecAdd::cleaner          },
    { "matrixMultiplication", mm::setup,               mm::launcher,              mm::cleaner              },
    { "sradv2_1",             sradv2_1::setup,         sradv2_1::launcher,        sradv2_1::cleaner        },
    { "sradv2_2",             sradv2_2::setup,         sradv2_2::launcher,        sradv2_2::cleaner        },
    { "nn",                   nn::setup,               nn::launcher,              nn::cleaner              },
    { "hotspot",              hotspot::setup,          hotspot::launcher,         hotspot::cleaner         },
    { "histogram",            histogram::setup,        histogram::launcher,       histogram::cleaner       },
    { "gaussian1",            gaussian1::setup,        gaussian1::launcher,       gaussian1::cleaner       },
    { "gaussian2",            gaussian2::setup,        gaussian2::launcher,       gaussian2::cleaner       },
    { "gemm",                 gemm::setup,             gemm::launcher,            gemm::cleaner            },
    { "blackScholes",         blackScholes::setup,     blackScholes::launcher,    blackScholes::cleaner    },
    { "binaryPartition",      binaryPartition::setup,  binaryPartition::launcher, binaryPartition::cleaner },
    { "backprop1",            backprop1::setup,        backprop1::launcher,       backprop1::cleaner       },
    { "backprop2",            backprop2::setup,        backprop2::launcher,       backprop2::cleaner       },
    { "bfs1",                 bfs1::setup,             bfs1::launcher,            bfs1::cleaner            },
    { "bfs2",                 bfs2::setup,             bfs2::launcher,            bfs2::cleaner            },
};

nvmlDevice_t device;
int num_kernels = sizeof(kernel_registry) / sizeof(kernel_registry[0]);

LauncherFn find_launcher(const char *name) {
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].launcher;
    fprintf(stderr, "ERROR: kernel launcher '%s' not found\n", name);
    return NULL;
}

CleanerFn find_cleaner(const char *name) {
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].cleaner;
    fprintf(stderr, "ERROR: kernel cleaner '%s' not found\n", name);
    return NULL;
}

SetupFn find_setup(const char *name) {
    for (int i = 0; i < num_kernels; i++)
        if (strcmp(kernel_registry[i].name, name) == 0)
            return kernel_registry[i].setup;
    fprintf(stderr, "ERROR: kernel setup '%s' not found\n", name);
    return NULL;
}

void warmup(int iterations) {
    const char *warmupKernel = "matrixMultiplication";
    SetupFn    warmupSetup    = find_setup(warmupKernel);
    LauncherFn warmupLauncher = find_launcher(warmupKernel);
    CleanerFn  warmupCleaner  = find_cleaner(warmupKernel);

    for (int i = 0; i < iterations; i++) {
        printf("\nWarmup run %d:\n", i + 1);
        warmupSetup(1, 0);
        cudaDeviceSynchronize();
        warmupLauncher(1, 0);
        warmupCleaner(0);
    }
}


typedef CUresult (*GreenCtxStreamCreateFn)(CUstream*, CUgreenCtx, unsigned int, int);

void createGreenContextStreams(int dev, int smsA, int smsB,
                               CUstream *streamA, CUstream *streamB,
                               CUgreenCtx *ctxA,  CUgreenCtx *ctxB,
                               GreenCtxStreamCreateFn cuGreenCtxStreamCreate)
{
    CUdevResource input_resources;
    CUdevResource remaining_resources;
    CUdevResourceDesc desc[2];

    checkCudaErrors(cuDeviceGetDevResource((CUdevice)dev, &input_resources, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int min_group_num_sms = input_resources.sm.minSmPartitionSize;
    unsigned int num_groups        = 999;

    checkCudaErrors(cuDevSmResourceSplitByCount(NULL, &num_groups, &input_resources,
                                                &remaining_resources, 0, min_group_num_sms));

    CUdevResource *target_resources = (CUdevResource*)calloc(num_groups, sizeof(CUdevResource));
    checkCudaErrors(cuDevSmResourceSplitByCount(target_resources, &num_groups, &input_resources,
                                                &remaining_resources, 0, min_group_num_sms));

    unsigned int groupsA = smsA / min_group_num_sms;
    unsigned int groupsB = smsB / min_group_num_sms;

    if ((groupsA + groupsB) > num_groups) {
        fprintf(stderr, "ERROR: SM split %d+%d exceeds device SMs (%u).\n",
                smsA, smsB, num_groups * min_group_num_sms);
        free(target_resources);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cuDevResourceGenerateDesc(&desc[0], &target_resources[0],       groupsA));
    checkCudaErrors(cuDevResourceGenerateDesc(&desc[1], &target_resources[groupsA], groupsB));

    checkCudaErrors(cuGreenCtxCreate(ctxA, desc[0], (CUdevice)dev, CU_GREEN_CTX_DEFAULT_STREAM));
    checkCudaErrors(cuGreenCtxCreate(ctxB, desc[1], (CUdevice)dev, CU_GREEN_CTX_DEFAULT_STREAM));

    checkCudaErrors(cuGreenCtxStreamCreate(streamA, *ctxA, CU_STREAM_NON_BLOCKING, 0));
    checkCudaErrors(cuGreenCtxStreamCreate(streamB, *ctxB, CU_STREAM_NON_BLOCKING, 0));

    free(target_resources);
}

void destroyGreenContextStreams(CUstream streamA, CUstream streamB,
                                CUgreenCtx ctxA,   CUgreenCtx ctxB)
{
    cuStreamDestroy(streamA);
    cuStreamDestroy(streamB);
    cuGreenCtxDestroy(ctxA);
    cuGreenCtxDestroy(ctxB);
}


void colocateKernels(const char *kernelA, const char *kernelB, int partitions_A, int partitions_B, 
                     cudaStream_t streamA, cudaStream_t streamB,
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

    cudaEventElapsedTime(&msTotal, globalStart, globalStop);

    float avgPowerW = ((powerBefore + powerAfter) / 2.0f) / 1000.0f;


    *time += msTotal;
    *power     += avgPowerW;
    *energy    += avgPowerW * (msTotal / 1000.0f);

    cleanerA(streamA);
        cudaError_t errA = cudaGetLastError();   
        if (errA != cudaSuccess)
            printf("*** Sticky error at start of (%s): %s\n", 
             kernelA, cudaGetErrorString(errA));
    cleanerB(streamB);
        cudaError_t errB = cudaGetLastError();   
        if (errB != cudaSuccess)
            printf("*** Sticky error at start of (%s): %s\n", 
             kernelB, cudaGetErrorString(errB));

    cudaEventDestroy(startA);   cudaEventDestroy(stopA);
    cudaEventDestroy(startB);   cudaEventDestroy(stopB);
    cudaEventDestroy(globalStart); cudaEventDestroy(globalStop);
    cudaStreamDestroy(syncStream);

}

int main(void)
{
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);
    checkCudaErrors(cuInit(0));

    int dev;
    cudaGetDevice(&dev);

    // Resolve cuGreenCtxStreamCreate once via driver API
    GreenCtxStreamCreateFn cuGreenCtxStreamCreate = nullptr;
    int cuda_driver_version;
    checkCudaErrors(cuDriverGetVersion(&cuda_driver_version));
    checkCudaErrors(cuGetProcAddress("cuGreenCtxStreamCreate",
                                     (void**)(&cuGreenCtxStreamCreate),
                                     cuda_driver_version, 0, nullptr));

    int iterations       = 10;
    const char *kernel_list_A[] = {"gemm","bfs1","blackScholes","nn","vecAdd"};
    const char *kernel_list_B[] = {"gaussian1","backprop2","gaussian1","backprop2","bfs2"};

    
    int sm_splitA[] = {40, 60, 20, 30, 50};
    int sm_splitB[] = {40, 20, 60, 50, 30};
   

    int nKernels    = sizeof(kernel_list_A)    / sizeof(kernel_list_A[0]);
    int nSplits     = sizeof(sm_splitA)      / sizeof(sm_splitA[0]);
    int nPairs      = nKernels * nSplits;

    PairResult *results = (PairResult*)malloc(nPairs * sizeof(PairResult));
    int resultIdx = 0;

    warmup(50);

    // ── Main experiment loop ──────────────────────────────────────────────────
    for (int i = 0; i < nKernels; i++) {
        const char *kernelA = kernel_list_A[i];
        const char *kernelB = kernel_list_B[i];
    
        int partitions_A = 1;
        int partitions_B = 1;
        for (int s = 0; s < nSplits; s++) {
            int smsA = sm_splitA[s];
            int smsB = sm_splitB[s];

            CUstream   streamA, streamB;
            CUgreenCtx ctxA,    ctxB;

            createGreenContextStreams(dev, smsA, smsB, &streamA, &streamB, &ctxA, &ctxB, cuGreenCtxStreamCreate);

            float accTimeTotal = 0.0f, accPower = 0.0f, accEnergy = 0.0f;

            printf("Kernel1 %-22s Kernel2 %-22s Partitions %d  SMs %d/%d\n",
                    kernelA, kernelB, partitions_A, smsA, smsB);

            for (int m = 0; m < iterations; m++) {
                colocateKernels(kernelA, kernelB,
                                partitions_A, partitions_B,
                                streamA, streamB, &accEnergy, &accPower, &accTimeTotal);
            }

            results[resultIdx++] = {
                kernelA, kernelB, partitions_A,
                smsA, smsB,
                accTimeTotal / iterations,
                accPower     / iterations,
                accEnergy    / iterations,
            };

            destroyGreenContextStreams(streamA, streamB, ctxA, ctxB);
        }
        
        
    }

    // ── Print results ─────────────────────────────────────────────────────────
    printf("\n%-22s %-22s %3s %6s %6s | %10s %10s %10s\n",
           "KernelA", "KernelB", "P", "smsA", "smsB",
           "Total(ms)", "Power(W)", "Energy(J)");
    printf("%-22s %-22s %3s %6s %6s | %10s %10s %10s\n",
           "----------------------", "----------------------", "---",
           "------", "------",
           "----------", "----------", "----------");

    for (int i = 0; i < resultIdx; i++) {
        printf("%-22s %-22s %3d %6d %6d | %10.3f %10.2f %10.4f\n",
               results[i].kernelA,      results[i].kernelB,
               results[i].partitions,   results[i].smsA,   results[i].smsB,
               results[i].avgTimeTotal, results[i].avgPower, results[i].avgEnergy);
    }

    free(results);
    nvmlShutdown();
    return 0;
}

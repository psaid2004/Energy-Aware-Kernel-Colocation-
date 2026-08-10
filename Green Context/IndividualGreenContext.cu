//Framework
#define CUDA_DRIVER_API         
// #include "../vecAdd/strippedVecAdd_temporal.cu"
// #include "../matrixMultiplication/strippedMatrixMultiplication_temporal.cu"
// #include "../srad_v2/strippedsradKernel1_temporal.cu"
// #include "../srad_v2/strippedsradKernel2_temporal.cu"
// #include "../nn/strippednn_temporal.cu"
// #include "../hotspot/strippedhotspot_temporal.cu"
// #include "../histogram/strippedHistogram_temporal.cpp"
// #include "../histogram/histogramKernel.cu"
// #include "../gaussian/strippedgaussian1_temporal.cu"
// #include "../gaussian/strippedgaussian2_temporal.cu"
// #include "../cudaTensorCoreGemm/strippedTensorCoreGemm_temporal.cu"
// #include "../BlackScholes/strippedBlackScholes_temporal.cu"
// #include "../binaryPartition/strippedBinaryPartition_temporal.cu"
// #include "../backprop/strippedbackprop1_temporal.cu"
// #include "../backprop/strippedbackprop2_temporal.cu"
// #include "../bfs/strippedbfs1_temporal.cu"
// #include "../bfs/strippedbfs2_temporal.cu"

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

// nvcc -std=c++17 greenContextIsolated.cu -I../Common -lcuda -lnvidia-ml -o greenContextIsolated
// CUDA_MODULE_LOADING=EAGER ./greenContextIsolated

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

typedef int (*SetupFn)(int partitions, cudaStream_t stream);
typedef int (*LauncherFn)(int partitions, cudaStream_t stream);
typedef int (*CleanerFn)(cudaStream_t stream);

typedef struct {
    const char *kernel;
    int sms;
    float avgTime;
    float avgPower;
    float avgEnergy;
} IsolatedResult;

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

void createSingleGreenContextStream(int dev, int sms,
                                     CUstream *stream, CUgreenCtx *ctx,
                                     GreenCtxStreamCreateFn cuGreenCtxStreamCreate)
{
    CUdevResource input_resources;
    CUdevResource remaining_resources;

    checkCudaErrors(cuDeviceGetDevResource((CUdevice)dev, &input_resources, CU_DEV_RESOURCE_TYPE_SM));

    unsigned int min_group_num_sms = input_resources.sm.minSmPartitionSize;
    unsigned int num_groups = 999;

    checkCudaErrors(cuDevSmResourceSplitByCount(NULL, &num_groups, &input_resources,
                                                &remaining_resources, 0, min_group_num_sms));

    CUdevResource *target_resources = (CUdevResource*)calloc(num_groups, sizeof(CUdevResource));
    checkCudaErrors(cuDevSmResourceSplitByCount(target_resources, &num_groups, &input_resources,
                                                &remaining_resources, 0, min_group_num_sms));

    unsigned int groups = sms / min_group_num_sms;

    if (groups > num_groups) {
        fprintf(stderr, "ERROR: SM count %d exceeds device SMs (%u).\n",
                sms, num_groups * min_group_num_sms);
        free(target_resources);
        exit(EXIT_FAILURE);
    }

    CUdevResourceDesc desc;
    checkCudaErrors(cuDevResourceGenerateDesc(&desc, &target_resources[0], groups));
    checkCudaErrors(cuGreenCtxCreate(ctx, desc, (CUdevice)dev, CU_GREEN_CTX_DEFAULT_STREAM));
    checkCudaErrors(cuGreenCtxStreamCreate(stream, *ctx, CU_STREAM_NON_BLOCKING, 0));

    free(target_resources);
}

void destroySingleGreenContextStream(CUstream stream, CUgreenCtx ctx)
{
    cuStreamDestroy(stream);
    cuGreenCtxDestroy(ctx);
}

void runIsolatedKernel(const char *kernelName, int partitions,
                       cudaStream_t stream,
                       float *energy, float *power, float *time)
{
    SetupFn    setup    = find_setup(kernelName);
    LauncherFn launcher = find_launcher(kernelName);
    CleanerFn  cleaner  = find_cleaner(kernelName);

    if (!launcher || !cleaner || !setup) {
        printf("Error finding kernel functions\n");
        exit(0);
    }

    setup(partitions, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    unsigned int powerBefore, powerAfter;
    nvmlDeviceGetPowerUsage(device, &powerBefore);

    cudaEventRecord(start, stream);
    launcher(partitions, stream);
    cudaEventRecord(stop, stream);

    cudaDeviceSynchronize();
    nvmlDeviceGetPowerUsage(device, &powerAfter);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float avgPowerW = ((powerBefore + powerAfter) / 2.0f) / 1000.0f;

    *time   += ms;
    *power  += avgPowerW;
    *energy += avgPowerW * (ms / 1000.0f);

    cleaner(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("*** Sticky error (%s): %s\n", kernelName, cudaGetErrorString(err));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(void)
{
    nvmlInit();
    nvmlDeviceGetHandleByIndex(0, &device);
    checkCudaErrors(cuInit(0));

    int dev;
    cudaGetDevice(&dev);

    GreenCtxStreamCreateFn cuGreenCtxStreamCreate = nullptr;
    int cuda_driver_version;
    checkCudaErrors(cuDriverGetVersion(&cuda_driver_version));
    checkCudaErrors(cuGetProcAddress("cuGreenCtxStreamCreate",
                                     (void**)(&cuGreenCtxStreamCreate),
                                     cuda_driver_version, 0, nullptr));

    int iterations = 10;

    const char *kernel_list[] = { "gaussian1"
    };

    int sm_counts[] = {20};

    int nKernels = sizeof(kernel_list) / sizeof(kernel_list[0]);
    int nSmCounts = sizeof(sm_counts) / sizeof(sm_counts[0]);
    int nResults = nKernels * nSmCounts;

    IsolatedResult *results = (IsolatedResult*)malloc(nResults * sizeof(IsolatedResult));
    int resultIdx = 0;

    warmup(50);

    for (int i = 0; i < nKernels; i++) {
        const char *kernelName = kernel_list[i];

        for (int s = 0; s < nSmCounts; s++) {
            int sms = sm_counts[s];

            CUstream stream;
            CUgreenCtx ctx;

            createSingleGreenContextStream(dev, sms, &stream, &ctx, cuGreenCtxStreamCreate);

            float accTime = 0.0f, accPower = 0.0f, accEnergy = 0.0f;

            printf("Kernel %-22s  SMs %d\n", kernelName, sms);

            for (int m = 0; m < iterations; m++) {
                runIsolatedKernel(kernelName, 1, stream,
                                  &accEnergy, &accPower, &accTime);
            }

            results[resultIdx++] = {
                kernelName, sms,
                accTime   / iterations,
                accPower  / iterations,
                accEnergy / iterations,
            };

            destroySingleGreenContextStream(stream, ctx);
        }
    }

    printf("\n%-22s %6s | %12s %12s %12s\n",
           "Kernel", "SMs", "AvgTime(ms)", "AvgPower(W)", "AvgEnergy(J)");
    printf("%-22s %6s | %12s %12s %12s\n",
           "----------------------", "------",
           "------------", "------------", "------------");

    for (int i = 0; i < resultIdx; i++) {
        printf("%-22s %6d | %12.3f %12.2f %12.4f\n",
               results[i].kernel, results[i].sms,
               results[i].avgTime, results[i].avgPower, results[i].avgEnergy);
    }

    free(results);
    nvmlShutdown();
    return 0;
}

// #include <assert.h>
// #include <stdio.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// // Helper functions and utilities to work with CUDA
// #define CUDA_DRIVER_API
// #include <helper_cuda.h>
// #include <helper_functions.h>

// /**
//  * Matrix multiplication (CUDA Kernel) on the device: C = A * B
//  * wA is A's width and wB is B's width
//  */
// template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
// {
// 	// Block index
// 	int bx = blockIdx.x;
// 	int by = blockIdx.y;

// 	// Thread index
// 	int tx = threadIdx.x;
// 	int ty = threadIdx.y;

// 	// Index of the first sub-matrix of A processed by the block
// 	int aBegin = wA * BLOCK_SIZE * by;

// 	// Index of the last sub-matrix of A processed by the block
// 	int aEnd = aBegin + wA - 1;

// 	// Step size used to iterate through the sub-matrices of A
// 	int aStep = BLOCK_SIZE;

// 	// Index of the first sub-matrix of B processed by the block
// 	int bBegin = BLOCK_SIZE * bx;

// 	// Step size used to iterate through the sub-matrices of B
// 	int bStep = BLOCK_SIZE * wB;

// 	// Csub is used to store the element of the block sub-matrix
// 	// that is computed by the thread
// 	float Csub = 0;

// 	// Loop over all the sub-matrices of A and B
// 	// required to compute the block sub-matrix
// 	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
// 		// Declaration of the shared memory array As used to
// 		// store the sub-matrix of A
// 		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

// 		// Declaration of the shared memory array Bs used to
// 		// store the sub-matrix of B
// 		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

// 		// Load the matrices from device memory
// 		// to shared memory; each thread loads
// 		// one element of each matrix
// 		As[ty][tx] = A[a + wA * ty + tx];
// 		Bs[ty][tx] = B[b + wB * ty + tx];

// 		// Synchronize to make sure the matrices are loaded
// 		__syncthreads();

// 		// Multiply the two matrices together;
// 		// each thread computes one element
// 		// of the block sub-matrix
// #pragma unroll

// 		for (int k = 0; k < BLOCK_SIZE; ++k) {
// 			Csub += As[ty][k] * Bs[k][tx];
// 		}

// 		// Synchronize to make sure that the preceding
// 		// computation is done before loading two new
// 		// sub-matrices of A and B in the next iteration
// 		__syncthreads();
// 	}

// 	// Write the block sub-matrix to device memory;
// 	// each thread writes one element
// 	int c               = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
// 	C[c + wB * ty + tx] = Csub;
// }

// void ConstantInit(float *data, int size, float val) {
// 	for (int i = 0; i < size; ++i) {
// 		data[i] = val;
// 	}
// }

// /**
//  * Run a simple test of matrix multiplication using CUDA
//  */
// int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA, const dim3 &dimsB, cudaStream_t stream) {
// 	// Allocate host memory for matrices A and B
// 	unsigned int size_A     = dimsA.x * dimsA.y;
// 	unsigned int mem_size_A = sizeof(float) * size_A;
// 	float       *h_A;
// 	h_A = (float *) malloc(mem_size_A);
// 	unsigned int size_B     = dimsB.x * dimsB.y;
// 	unsigned int mem_size_B = sizeof(float) * size_B;
// 	float       *h_B;
// 	h_B = (float *) malloc(mem_size_B);
// 	// cudaStream_t stream;

// 	printf("Malloc'd host memory\n");

// 	// Initialize host memory
// 	const float valB = 0.01f;
// 	ConstantInit(h_A, size_A, 1.0f);
// 	ConstantInit(h_B, size_B, valB);

// 	// Allocate device memory
// 	float *d_A, *d_B, *d_C;

// 	// Allocate host matrix C
// 	dim3         dimsC(dimsB.x, dimsA.y, 1);
// 	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
// 	float       *h_C;
// 	h_C = (float *) malloc(mem_size_C);

// 	if (h_C == NULL) {
// 		fprintf(stderr, "Failed to allocate host matrix C!\n");
// 		exit(EXIT_FAILURE);
// 	}

// 	checkCudaErrors(cudaMalloc((void **) (&d_A), mem_size_A));
// 	checkCudaErrors(cudaMalloc((void **) (&d_B), mem_size_B));
// 	checkCudaErrors(cudaMalloc((void **) (&d_C), mem_size_C));
// 	// Allocate CUDA events that we'll use for timing
// 	cudaEvent_t start, stop;
// 	checkCudaErrors(cudaEventCreate(&start));
// 	checkCudaErrors(cudaEventCreate(&stop));

// 	// checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

// 	// copy host memory to device
// 	checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
// 	checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

// 	// Setup execution parameters
// 	dim3 threads(block_size, block_size); // 2D thread block containing block_size X block_size threads.
// 	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y); // 2D grid containing (dimsB.x / threads.x) X (dimsA.y / threads.y) blocks.
// 	// Total threads = (grid.x X grid.y) x (threads.x X threads.y)

// 	// Use iterations to get more precise timings.
// 	int iterations = 20;

// 	// Create and start timer
// 	printf("Computing result using CUDA Kernel...\n");

// 	checkCudaErrors(cudaEventRecord(start, stream));
// 	checkCudaErrors(cudaEventSynchronize(start));

// 	for (int j = 0; j < iterations; j++) {
// 		if (block_size == 16) {
// 			MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
// 		} else {
// 			MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
// 		}
// 	}

// 	checkCudaErrors(cudaEventRecord(stop, stream));
// 	checkCudaErrors(cudaEventSynchronize(stop));

// 	float msecTotal = 0.0f;
// 	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

// 	// Compute and print the performance
// 	float  msecPerMatrixMul = msecTotal / iterations;
// 	double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) * static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
// 	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
// 	printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
// 		   " WorkgroupSize= %u threads/block\n",
// 		   gigaFlops,
// 		   msecPerMatrixMul,
// 		   flopsPerMatrixMul,
// 		   threads.x * threads.y);

// 	// Copy result from device to host
// 	checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
// 	checkCudaErrors(cudaStreamSynchronize(stream)); // Note: a stream synchronize here makes the previous "Async" call synchronous/blocking (so making it async is useless).

// 	printf("Checking computed result for correctness: ");
// 	bool correct = true;

// 	// test relative error by the formula
// 	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
// 	double eps = 1.e-6; // machine zero

// 	for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
// 		double abs_err    = fabs(h_C[i] - (dimsA.x * valB));
// 		double dot_length = dimsA.x;
// 		double abs_val    = fabs(h_C[i]);
// 		double rel_err    = abs_err / abs_val / dot_length;

// 		if (rel_err > eps) {
// 			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x * valB, eps);
// 			correct = false;
// 		}
// 	}

// 	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

// 	// Clean up memory
// 	free(h_A);
// 	free(h_B);
// 	free(h_C);
// 	checkCudaErrors(cudaFree(d_A));
// 	checkCudaErrors(cudaFree(d_B));
// 	checkCudaErrors(cudaFree(d_C));
// 	checkCudaErrors(cudaEventDestroy(start));
// 	checkCudaErrors(cudaEventDestroy(stop));

// 	if (correct) {
// 		return EXIT_SUCCESS;
// 	}
// 	else {
// 		return EXIT_FAILURE;
// 	}
// }


// int main(int argc, char **argv) {
// 	checkCudaErrors(cuInit(0));
// 	int dev = findCudaDevice(argc, (const char **)argv);

// 	int block_size = 32;

// 	// Default dimenions of matrices.
// 	// dim3 dimsA(8192, 8192, 1);
// 	// dim3 dimsB(8192, 8192, 1);
// 	dim3 dimsA(4096, 4096, 1);
// 	dim3 dimsB(4096, 4096, 1);

// 	int stream1_target_sms = getCmdLineArgumentInt(argc, (const char **)argv, "stream1-sms"); // Default = 0
// 	int stream2_target_sms = getCmdLineArgumentInt(argc, (const char **)argv, "stream2-sms"); // Default = 0

// 	if (dimsA.x != dimsB.y) {
// 		printf("Error: outer matrix dimensions must be equal. (%d != %d)\n", dimsA.x, dimsB.y);
// 		exit(EXIT_FAILURE);
// 	}

// 	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

// 	CUstream stream1, stream2;
// 	CUgreenCtx contexts[2];
//   CUdevResourceDesc desc[2];
//   CUdevResource input_resources;
//   CUdevResource *target_resources;
//   CUdevResource remaining_resources[2];

// 	// CUresult(CUDAAPI*)(CUstream*, CUgreenCtx, unsigned int, int) ;
// 	CUresult (*cuGreenCtxStreamCreate)(CUstream*, CUgreenCtx, unsigned int, int);

// 	int cuda_driver_version;
// 	checkCudaErrors(cuDriverGetVersion(&cuda_driver_version));
// 	checkCudaErrors(cuGetProcAddress("cuGreenCtxStreamCreate", (void **) (&cuGreenCtxStreamCreate), cuda_driver_version, 0, nullptr));

// 	checkCudaErrors(cuDeviceGetDevResource((CUdevice) dev, &input_resources, CU_DEV_RESOURCE_TYPE_SM));

// 	printf("input_resources.sm.minSmPartitionSize: %u\n", input_resources.sm.minSmPartitionSize);

// 	unsigned int num_groups = 999;
// 	unsigned int min_group_num_sms = input_resources.sm.minSmPartitionSize;

// 	if (stream1_target_sms < min_group_num_sms || stream2_target_sms < min_group_num_sms) {
// 		fprintf(stderr, "ERROR: Requested SMs less than minSmPartitionSize (%u).\n", min_group_num_sms);
// 		exit(EXIT_FAILURE);
// 	}

// 	// Simulate execution by passing NULL as the target resources array. Sets num_target_groups.
// 	checkCudaErrors(cuDevSmResourceSplitByCount(NULL, &num_groups, &input_resources, &remaining_resources[0], 0, min_group_num_sms));
// 	target_resources = (CUdevResource *) calloc(num_groups, sizeof(CUdevResource));

// 	// Creates min_group_num_sms x num_groups CUdevResource's in target_resources.
// 	checkCudaErrors(cuDevSmResourceSplitByCount(target_resources, &num_groups, &input_resources, &remaining_resources[0], 0, min_group_num_sms));

// 	// Round down to lowest min_group_num_sms multiple.
// 	unsigned int stream1_num_groups = stream1_target_sms / min_group_num_sms;
// 	unsigned int stream2_num_groups = stream2_target_sms / min_group_num_sms;

// 	if ((stream1_num_groups + stream2_num_groups) > num_groups) {
// 		fprintf(stderr, "ERROR: Requested SM split exceeds number of SMs on device.\n");
// 		exit(EXIT_FAILURE);
// 	}

// 	checkCudaErrors(cuDevResourceGenerateDesc(&desc[0], &target_resources[0], stream1_num_groups));
// 	checkCudaErrors(cuDevResourceGenerateDesc(&desc[1], &target_resources[stream1_num_groups], stream2_num_groups));

// 	printf("Stream 1 number of sms: %u\n", stream1_num_groups * min_group_num_sms);
// 	printf("Stream 2 number of sms: %u\n", stream2_num_groups * min_group_num_sms);
// 	printf("Unused number of sms: %u\n", remaining_resources[0].sm.smCount + (num_groups - stream1_num_groups - stream2_num_groups) * min_group_num_sms);

// 	checkCudaErrors(cuGreenCtxCreate(&contexts[0], desc[0], (CUdevice) dev, CU_GREEN_CTX_DEFAULT_STREAM));
// 	checkCudaErrors(cuGreenCtxCreate(&contexts[1], desc[1], (CUdevice) dev, CU_GREEN_CTX_DEFAULT_STREAM));

// 	checkCudaErrors(cuGreenCtxStreamCreate(&stream1, contexts[0], CU_STREAM_NON_BLOCKING, 0));
// 	checkCudaErrors(cuGreenCtxStreamCreate(&stream2, contexts[1], CU_STREAM_NON_BLOCKING, 0));

// 	int matrix_result1 = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, stream1);
// 	int matrix_result2 = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, stream2);

// 	exit(matrix_result1 || matrix_result2);
// }

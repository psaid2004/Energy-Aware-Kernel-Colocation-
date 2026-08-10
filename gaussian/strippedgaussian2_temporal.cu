/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>

// #include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 256
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 32
#endif

// #define NUM_OF_PARTITIONS 2


namespace gaussian2{
static int Size;
static float *a, *b, *m, *m_cuda,*a_cuda, *b_cuda;
static int block_size,grid_size;
static int chunkRows,chunkSize;
static dim3 dimBlock2, dimGrid2;
static int launches;

FILE *fp;

void ForwardSub();
__global__ void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
void InitAry(float *ary, int ary_size);
void PrintAry(float *ary, int ary_size);

// unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float* coe = new float[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t, int row_offset, int col_offset)
{
  #pragma unroll 1
     for (int iterations = 0; iterations <180000; iterations++){

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y+col_offset;

    // global row
    int global_xidx = (xidx + row_offset)%Size;

  if(global_xidx + 1 + t >= Size) {
      // printf("RETURNED global_xidx: %d, t: %d, Size: %d\n", global_xidx, t, Size);
      return;
    }
    if(yidx >= Size - t) {
      // printf("RETURNED yidx: %d, t: %d, Size: %d\n", yidx, t, Size);
      return;
    }

    a_cuda[Size*(global_xidx+1+t)+(yidx+t)] -= m_cuda[Size*(global_xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];

    if(yidx == 0){

        b_cuda[global_xidx+1+t] -= m_cuda[Size*(global_xidx+1+t)+t] * b_cuda[t];
    }
  }
}


/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2f ", ary[i]);
	}
	printf("\n\n");
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}


int setup(int partitions, cudaStream_t stream){
    // printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    srand(42);
    Size = 192; // Causes total 65536 threads
    // Size = 40;
     int size =Size;


    cudaMallocHost(&a, Size * Size * sizeof(float));
    cudaMallocHost(&b, Size * sizeof(float));
    cudaMallocHost(&m, Size * Size * sizeof(float));

    
    create_matrix(a, Size);

    for (int j =0; j< size; j++)
  	b[j]=1.0;
    
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            m[i * Size + j] =
                (float)rand() / (float)RAND_MAX;
        }
    }
	
    // allocate memory on GPU
    cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
    
    cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
    cudaMalloc((void **) &b_cuda, Size * sizeof(float));

	

    dimBlock2 = dim3(BLOCK_SIZE_XY, BLOCK_SIZE_XY); 
    dimGrid2  = dim3(15, 16/partitions);                     
  
    launches      = partitions;               
    chunkRows = Size / launches;           

    printf("Launched %d kernels with: %dx%d blocks, Threads: %d\n", launches, dimGrid2.x, dimGrid2.y, dimBlock2.x * dimBlock2.y);

    cudaMemcpyAsync(m_cuda, m, Size*Size*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(a_cuda, a, Size*Size*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b_cuda, b, Size*sizeof(float),      cudaMemcpyHostToDevice, stream);
    return 0;
}


int launcher(int partitions, cudaStream_t stream)
{

		// Fan2<<<1, 1, 0, stream>>>(m_cuda, a_cuda, b_cuda, Size, Size, 0, 0);

	for (int i = 0; i < launches; i++) {
		int row_offset = i * chunkRows;

		Fan2<<<dimGrid2, dimBlock2, 0, stream>>>(m_cuda, a_cuda, b_cuda, Size, Size, 0, row_offset, dimGrid2.y * i);

	}

  return 0;
}

int cleaner(cudaStream_t stream){
  cudaMemcpyAsync(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost, stream);
cudaDeviceSynchronize();

	// PrintAry(b, Size);
  cudaFree(m_cuda);
 	cudaFree(a_cuda);
  	cudaFreeHost(m);
  	cudaFreeHost(a);
  	cudaFreeHost(b);
    return 0;
}
}
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


//run this code using the framework

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

namespace gaussian1{
static int Size;
static float *a, *b, *m, *m_cuda,*a_cuda;
static int block_size,grid_size;
static int chunkRows,chunkSize;
static dim3 dimBlock, dimGrid;
static int launches;



FILE *fp;

void InitPerRun();
__global__ void Fan1(float *m, float *a, int Size, int t);
void InitMat(float *ary, int nrow, int ncol);
void PrintMat(float *ary, int nrow, int ncolumn);
void checkCUDAError(const char *msg);

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

 


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(float* m) 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;

}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t, int offset)
{   
  #pragma unroll 1
  for (int iterations = 0; iterations <430000; iterations++){

	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);
  int current_row = (threadIdx.x + blockIdx.x * blockDim.x + t + 1 + offset) % Size; // Modulo to keep threads from being idle, impacts correctness of output
  // printf("%d\n", current_row);

	if(current_row >= Size) return;
	int local_row = current_row - offset;
  m_cuda[Size * local_row + t] = a_cuda[Size * current_row + t] / a_cuda[Size*t+t];
  }
}



int setup(int partitions, cudaStream_t stream){
    printf("in setup function for gaussian1\n");

    Size = 7680*4;
  
    cudaMallocHost(&a, Size * Size * sizeof(float));
    cudaMallocHost(&b, Size * sizeof(float));
    cudaMallocHost(&m, Size * Size * sizeof(float));


    create_matrix(a, Size);

    for (int j =0; j< Size; j++)
  	b[j]=1.0;
    InitPerRun(m);

	
	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	
	
	int blocksPerLaunch =80;
	block_size = 32;

  dimBlock   = dim3(block_size);  
  dimGrid    = dim3(blocksPerLaunch);         
  launches   = (Size/ block_size)/blocksPerLaunch;                

  chunkRows  = Size / launches;   
  chunkSize  = chunkRows * Size;

printf("Launched %d kernels with: %d blocks, Threads: %d\n", launches, dimGrid.x, block_size);

  cudaMemcpyAsync(a_cuda, a, Size*Size*sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(m_cuda, m, Size*Size*sizeof(float), cudaMemcpyHostToDevice, stream);
  return 0;

}


int launcher(int partitions, cudaStream_t stream){

	for (int i =0; i < launches; i++) {
    int row_offset = i * chunkRows;
    int offset_m = i * chunkSize;
		
		Fan1<<<dimGrid,dimBlock,0, stream>>>(m_cuda + offset_m,a_cuda, Size,0, row_offset);
	}

  return 0;
}

int cleaner(cudaStream_t stream){

  // PrintMat(m, Size, Size);
  cudaMemcpyAsync(m, m_cuda, Size*Size*sizeof(float), cudaMemcpyDeviceToHost, stream);

cudaDeviceSynchronize();
  cudaFree(m_cuda);
	cudaFree(a_cuda);
  cudaFreeHost(m);
  cudaFreeHost(a);
  cudaFreeHost(b);
  return 0;
}

void InitMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+Size*i+j);
		}
	}  
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}
}
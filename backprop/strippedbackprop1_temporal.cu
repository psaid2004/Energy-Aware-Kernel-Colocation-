// This file should be ran using the framework


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
// #include <sys/time.h>

// includes, kernels
#include "backprop_cuda_kernel.cu"
#include "backprop.h"





extern "C"
void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern "C"
void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern "C"
void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern "C" 
void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);



extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);


namespace backprop1{
  ////////////////////////////////////////////////////////////////////////////////

static float *alloc_1d(int n) {
    float *p = (float*)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "alloc_1d failed\n"); exit(1); }
    return p;
}

static float **alloc_2d(int m, int n) {
    float **p = (float**)malloc(m * sizeof(float*));
    if (!p) { fprintf(stderr, "alloc_2d failed\n"); exit(1); }
    for (int i = 0; i < m; i++) p[i] = alloc_1d(n);
    return p;
}

static BPNN *bpnn_create(int n_in, int n_hidden, int n_out) {
    BPNN *b = (BPNN*)malloc(sizeof(BPNN));
    b->input_n  = n_in;
    b->hidden_n = n_hidden;
    b->output_n = n_out;
    b->input_units  = alloc_1d(n_in + 1);
    b->hidden_units = alloc_1d(n_hidden + 1);
    b->output_units = alloc_1d(n_out + 1);
    b->hidden_delta = alloc_1d(n_hidden + 1);
    b->output_delta = alloc_1d(n_out + 1);
    b->target       = alloc_1d(n_out + 1);
    b->input_weights       = alloc_2d(n_in + 1, n_hidden + 1);
    b->hidden_weights      = alloc_2d(n_hidden + 1, n_out + 1);
    b->input_prev_weights  = alloc_2d(n_in + 1, n_hidden + 1);
    b->hidden_prev_weights = alloc_2d(n_hidden + 1, n_out + 1);
    // randomize weights
    for (int i = 0; i <= n_in; i++)
        for (int j = 0; j <= n_hidden; j++) {
            b->input_weights[i][j]      = (float)rand() / RAND_MAX;
            b->input_prev_weights[i][j] = 0.0f;
        }
    for (int i = 0; i <= n_hidden; i++)
        for (int j = 0; j <= n_out; j++) {
            b->hidden_weights[i][j]      = (float)rand() / RAND_MAX;
            b->hidden_prev_weights[i][j] = 0.0f;
        }
    // randomize target
    for (int i = 0; i <= n_out; i++) b->target[i] = 0.1f;
    return b;
}

static void bpnn_free(BPNN *b) {
    if (!b) return;
    int n1 = b->input_n, n2 = b->hidden_n;
    free(b->input_units);  free(b->hidden_units); free(b->output_units);
    free(b->hidden_delta); free(b->output_delta); free(b->target);
    for (int i = 0; i <= n1; i++) {
        free(b->input_weights[i]);
        free(b->input_prev_weights[i]);
    }
    free(b->input_weights); free(b->input_prev_weights);
    for (int i = 0; i <= n2; i++) {
        free(b->hidden_weights[i]);
        free(b->hidden_prev_weights[i]);
    }
    free(b->hidden_weights); free(b->hidden_prev_weights);
    free(b);
}

static void load(BPNN *b) {
    for (int i = 1; i <= b->input_n; i++)
        b->input_units[i] = (float)rand() / RAND_MAX;
}

  static unsigned int num_threads = 0;
  static unsigned int num_blocks = 0;
  static   int in, hid, out, seed, layer_size;
  static int m = 0;
  static float *input_hidden_cuda, *input_cuda, *output_hidden_cuda, *partial_sum, *hidden_partial_sum, *input_weights_one_dim;
  static float sum;
  static int blocksPerPartition;
  static dim3  grid, threads; 
  static int inputRowsPerPartition, chunkSize_input_weights, chunkSize_partial;
  static BPNN *net;
  static float out_err, hid_err;
  static float *input_weights_prev_one_dim;
  static int launches;





////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////


int setup(int partitions, cudaStream_t stream){
	

  layer_size = 3840;
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }
  

  seed = 7;   
  srand(seed);
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  load(net);


  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;   
   
  int m = 0;

  num_blocks = in / 16;  

  

  cudaMallocHost(&input_weights_one_dim, (in + 1)* (hid + 1) * sizeof(float));
  cudaMallocHost(&input_weights_prev_one_dim, (in + 1)* (hid + 1) * sizeof(float));
  cudaMallocHost(&partial_sum, num_blocks * WIDTH * sizeof(float));



  // this preprocessing stage is added to correct the bugs of wrong memcopy using two-dimensional net->inputweights
  for (int k = 0; k <= in; k++) {	
   for (int j = 0; j <= hid; j++) {
	  input_weights_one_dim[m] = net->input_weights[k][j];
	  input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	  m++;
    }
  }
  
  cudaMalloc((void**) &input_cuda, (in + 1) * sizeof(float));
  cudaMalloc((void**) &output_hidden_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float));
  cudaMalloc((void**) &hidden_partial_sum, num_blocks * WIDTH * sizeof(float));

  int blocksPerLaunch = 240/partitions;
  launches = num_blocks / blocksPerLaunch;  
  grid = dim3( 1 , blocksPerLaunch, 1);
  threads = dim3(16 , 16); 
  printf("Lauched %d kernels, of %d blocks, with %d threads\n", launches, grid.y, threads.x*threads.y);

//   printf("Performing GPU computation\n");
  inputRowsPerPartition = (in / launches); 
  chunkSize_input_weights = inputRowsPerPartition * (hid+1); 
  chunkSize_partial = blocksPerLaunch * WIDTH;
  cudaMemcpyAsync(input_cuda, net->input_units, (in+1)*sizeof(float), cudaMemcpyHostToDevice, stream); 
  cudaMemcpyAsync(input_hidden_cuda, input_weights_one_dim, (in+1) * (hid+1)  * sizeof(float),
                    cudaMemcpyHostToDevice, stream);               
    return 0;
}

int launcher(int partitions, cudaStream_t stream)
{  
 
    
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
    for (int i = 0; i < launches; i++){
        int offset_input   = i * inputRowsPerPartition;    
        int offset_weights = i * chunkSize_input_weights; 
        int offset_partial = i * chunkSize_partial;       

        
        
        bpnn_layerforward_CUDA<<< grid, threads, 0, stream >>>(input_cuda,
                                                    output_hidden_cuda,
                                                    input_hidden_cuda + offset_weights,
                                                    hidden_partial_sum + offset_partial,
                                                    grid.y*i,
                                                    hid);
                
        cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("bpnn kernel error: %s\n", cudaGetErrorString(error));
                exit(EXIT_FAILURE);
            }
        
        }

return 0;

}

int cleaner(cudaStream_t stream) {
cudaMemcpyAsync(partial_sum, hidden_partial_sum,  num_blocks * WIDTH * sizeof(float), cudaMemcpyDeviceToHost, stream);

cudaDeviceSynchronize();
  bpnn_free(net);
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);  
  cudaFreeHost(partial_sum);
  cudaFreeHost(input_weights_one_dim);
  cudaFreeHost(input_weights_prev_one_dim);
  return 0;
}

}
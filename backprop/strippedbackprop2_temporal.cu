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



////////////////////////////////////////////////////////////////////////////////





extern "C"
float **alloc_2d_dbl(int m, int n);

extern "C"
float squash(float x);


namespace backprop2{

  static void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2) {
    float sum;
    l1[0] = 1.0f;
    for (int j = 1; j <= n2; j++) {
        sum = 0.0f;
        for (int k = 0; k <= n1; k++)
            sum += conn[k][j] * l1[k];
        l2[j] = 1.0f / (1.0f + expf(-sum));
    }
}

static void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err) {
    float errsum = 0.0f;
    for (int j = 1; j <= nj; j++) {
        float o = output[j];
        float t = target[j];
        delta[j] = o * (1.0f - o) * (t - o);
        errsum += fabsf(delta[j]);
    }
    *err = errsum;
}

static void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no,
                               float **who, float *hidden, float *err) {
    float errsum = 0.0f;
    for (int j = 1; j <= nh; j++) {
        float h = hidden[j];
        float sum = 0.0f;
        for (int k = 1; k <= no; k++)
            sum += delta_o[k] * who[j][k];
        delta_h[j] = h * (1.0f - h) * sum;
        errsum += fabsf(delta_h[j]);
    }
    *err = errsum;
}

static void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly,
                                 float **w, float **oldw) {
    ly[0] = 1.0f;
    for (int j = 1; j <= ndelta; j++) {
        for (int k = 0; k <= nly; k++) {
            float new_dw = (ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]);
            w[k][j]   += new_dw;
            oldw[k][j] = new_dw;
        }
    }
}

  
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
  static float *input_hidden_cuda, *input_cuda, *output_hidden_cuda, *partial_sum, *hidden_partial_sum, *input_weights_one_dim, *input_weights_prev_one_dim, *hidden_delta_cuda, *input_prev_weights_cuda;
  static float sum;
  static int blocksPerPartition;
  static dim3  grid, threads; 
  static int inputRowsPerPartition, chunkSize_weights, chunkSize_input;
  static BPNN *net;
  static float out_err, hid_err;
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


  bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);



  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
  bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);


  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);


  cudaMalloc((void**) &hidden_delta_cuda, (hid + 1) * sizeof(float));
  cudaMalloc((void**) &input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float));

  
  int blocksPerLaunch = 240/partitions;
  launches = num_blocks / blocksPerLaunch;  
  grid = dim3( 1 , blocksPerLaunch, 1);
  threads = dim3(16 , 16); 
  printf("Lauched %d kernels, of %d blocks, with %d threads\n", launches, grid.y, threads.x*threads.y);

  // printf("Performing GPU computation\n");
  chunkSize_weights = blocksPerLaunch * HEIGHT * (hid + 1);
  chunkSize_input = blocksPerLaunch * HEIGHT;
    printf("Lauched %d kernels, of %d blocks, with %d threads\n", launches, grid.y, threads.x*threads.y);

  cudaMemcpyAsync(hidden_delta_cuda, net->hidden_delta, (hid+1)*sizeof(float), cudaMemcpyHostToDevice, stream);
 
  cudaMemcpyAsync(input_hidden_cuda, input_weights_one_dim, num_blocks*HEIGHT * (hid + 1)* sizeof(float), cudaMemcpyHostToDevice, stream);

  cudaMemcpyAsync(input_prev_weights_cuda, input_weights_prev_one_dim, num_blocks*HEIGHT * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(input_cuda, net->input_units, num_blocks*HEIGHT * sizeof(float), cudaMemcpyHostToDevice, stream);
    

    return 0;
}

int launcher(int partitions, cudaStream_t stream)
{  
  //printf("in= %d, hid = %d, numblocks = %d\n", in, hid, num_blocks);
for (int i =0; i < launches; i++){
    int offset_weights = i * chunkSize_weights;
    int offset_input   = i * chunkSize_input;

    bpnn_adjust_weights_cuda<<< grid, threads, 0, stream >>>(hidden_delta_cuda,  
                                                    hid, 
                                                    input_cuda + offset_input, 
                                                    i,
                                                    input_hidden_cuda + offset_weights, 
                                                    input_prev_weights_cuda + offset_weights
                                                    );

 
  }
return 0;

}

int cleaner(cudaStream_t stream) {

    
cudaMemcpyAsync(net->input_units, input_cuda, num_blocks*HEIGHT * sizeof(float), cudaMemcpyDeviceToHost, stream);

cudaMemcpyAsync(input_weights_one_dim, input_hidden_cuda, num_blocks*HEIGHT * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost, stream);
cudaDeviceSynchronize();
  cudaFree(input_cuda);
  cudaFree(output_hidden_cuda);
  cudaFree(input_hidden_cuda);
  cudaFree(hidden_partial_sum);
  cudaFree(input_prev_weights_cuda);
  cudaFree(hidden_delta_cuda);
  
  cudaFreeHost(partial_sum);
  cudaFreeHost(input_weights_one_dim);
  cudaFreeHost(input_weights_prev_one_dim);

  return 0;
}

}
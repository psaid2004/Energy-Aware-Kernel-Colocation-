/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/


// This file should be ran using the framework


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "kernel2.cu"

#define MAX_THREADS_PER_BLOCK 512
#define BFS_INPUT_FILE "../bfs/data/graph65536.txt"

namespace bfs2 {

static int    no_of_nodes;
static int    edge_list_size;
static int    chunkSize;

static Node  *d_graph_nodes;
static int   *d_graph_edges;
static bool  *d_graph_mask;
static bool  *d_updating_graph_mask;
static bool  *d_graph_visited;
static int   *d_cost;
static bool  *d_over;

static bool  *h_graph_mask;
static bool  *h_updating_graph_mask;
static bool  *h_graph_visited;
static int   *h_cost;

static dim3   grid, threads;
static int    num_of_blocks;
static int    num_of_threads_per_block;
static Node *h_graph_nodes;
static int *h_graph_edges;
static bool stop = false;
static int launches;




int setup(int partitions, cudaStream_t stream) {
    // Read file and allocate exactly like bfs1
FILE *fp = fopen(BFS_INPUT_FILE, "r");
    if (!fp) { printf("Error Reading graph file\n"); return 1; }

    fscanf(fp, "%d", &no_of_nodes);


    num_of_blocks = 1;
    num_of_threads_per_block = no_of_nodes;
    if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }

    int blocksPerLaunch = 120/partitions;
    grid    = dim3(blocksPerLaunch, 1, 1);
    launches = partitions;
    chunkSize = no_of_nodes / launches;
    printf("num of nodes %d", no_of_nodes);
    printf("number of blocks: %d, threads per block: %d\n", num_of_blocks, num_of_threads_per_block);

    threads = dim3(num_of_threads_per_block, 1, 1);
    printf("Lauched %d kernels, of %d blocks, with %d threads\n", launches, grid.x, threads.x);

    cudaMallocHost(&h_graph_mask,           no_of_nodes * sizeof(bool));
    cudaMallocHost(&h_updating_graph_mask,  no_of_nodes * sizeof(bool));
    cudaMallocHost(&h_graph_visited,        no_of_nodes * sizeof(bool));
    cudaMallocHost(&h_cost,                 no_of_nodes * sizeof(int));

    cudaMallocHost(&h_graph_nodes, no_of_nodes * sizeof(Node));

    int start, edgeno;
    for (unsigned int i = 0; i < no_of_nodes; i++) {
        fscanf(fp, "%d %d", &start, &edgeno);
        h_graph_nodes[i].starting    = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i]             = false;
        h_updating_graph_mask[i]    = false;
        h_graph_visited[i]          = false;
        h_cost[i]                   = -1;
    }
    for (int i = 0; i < no_of_nodes; i += 25) {
    // h_updating_graph_mask[i] = true;  
    }

    int source = 0;
    fscanf(fp, "%d", &source);
    source = 0;
    h_graph_mask[source]    = true;
    h_graph_visited[source] = true;
    h_cost[source]          = 0;

    fscanf(fp, "%d", &edge_list_size);
    cudaMallocHost(&h_graph_edges, edge_list_size * sizeof(int));
    for (int i = 0; i < edge_list_size; i++) {
        int id, cost;
        fscanf(fp, "%d %d", &id, &cost);
        h_graph_edges[i] = id;
    }
    fclose(fp);

    // Device allocations

    cudaMalloc((void**) &d_graph_mask,           sizeof(bool) * no_of_nodes);
    cudaMalloc((void**) &d_updating_graph_mask,  sizeof(bool) * no_of_nodes);
    cudaMalloc((void**) &d_graph_visited,        sizeof(bool) * no_of_nodes);
    cudaMalloc((void**) &d_over,                 sizeof(bool));

    cudaMemcpyAsync(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice, stream);

   


    return 0;
}

int launcher(int partitions, cudaStream_t stream) {

    for (int i = 0; i < launches; i++) {
        int offset = i * chunkSize;
        Kernel2<<<grid, threads, 0, stream>>>(
            d_graph_mask          + offset,
            d_updating_graph_mask + offset,
            d_graph_visited       + offset,
            d_over,
            chunkSize);
    }

    return 0;
}

int cleaner(cudaStream_t stream) {
    cudaMemcpyAsync(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();

    // // Print results
    // int visited_count = 0;
    // int frontier_count = 0;
    // for (int i = 0; i < no_of_nodes; i++) {
    //     if (h_graph_visited[i]) visited_count++;
    //     if (h_graph_mask[i])    frontier_count++;
    // }
    // printf("BFS Kernel2 results:\n");
    // printf("  Nodes visited:       %d / %d\n", visited_count,  no_of_nodes);
    // printf("  Nodes in frontier:   %d / %d\n", frontier_count, no_of_nodes);

    cudaFree(d_graph_mask);
    cudaFree(d_updating_graph_mask);
    cudaFree(d_graph_visited);
    cudaFree(d_cost);
    cudaFree(d_over);
    cudaFreeHost(h_graph_mask);
    cudaFreeHost(h_updating_graph_mask);
    cudaFreeHost(h_graph_visited);
    cudaFreeHost(h_cost);
    cudaFreeHost(h_graph_nodes);
    cudaFreeHost(h_graph_edges);
    return 0;
}

} 
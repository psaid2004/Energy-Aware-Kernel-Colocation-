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
#define BFS_INPUT_FILE "../bfs/data/graph65536.txt"

#define MAX_THREADS_PER_BLOCK 512

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};
#include "kernel.cu"

namespace bfs1{

static int no_of_nodes, edge_list_size, num_of_blocks, num_of_threads_per_block, chunkSize;
static FILE *fp;
static Node* h_graph_nodes;
static bool *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
static int id,cost;
static int* h_graph_edges;
	//Allocate memory for Device variables
static Node* d_graph_nodes;
static int *d_graph_edges;
static bool *d_graph_mask, *d_updating_graph_mask, *d_graph_visited, *d_over;

static int *d_cost, *h_cost;

static	dim3  grid, threads;
static int launches;




void BFSGraph(int argc, char** argv);


int setup(int partitions, cudaStream_t stream) {

FILE *fp = fopen(BFS_INPUT_FILE, "r");
    if (!fp) { printf("Error Reading graph file\n"); return 1; }
	no_of_nodes=0;
	edge_list_size=0;


int source = 0;

	fscanf(fp,"%d",&no_of_nodes);
	// printf("%d\n",no_of_nodes);

	num_of_blocks = 1;
	num_of_threads_per_block = no_of_nodes;

	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
		num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}


	printf("number of blocks: %d, threads per block: %d\n", num_of_blocks, num_of_threads_per_block);
	int blocksPerLaunch = 40;
    launches = num_of_blocks/blocksPerLaunch;
	chunkSize = no_of_nodes/launches;


    cudaMallocHost(&h_graph_nodes, no_of_nodes * sizeof(Node));
	cudaMallocHost(&h_graph_mask, no_of_nodes * sizeof(bool));
	cudaMallocHost(&h_updating_graph_mask, no_of_nodes * sizeof(bool));
	cudaMallocHost(&h_graph_visited, no_of_nodes * sizeof(bool));



	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	
	cudaMallocHost(&h_graph_edges, edge_list_size * sizeof(int));

	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
		fclose(fp);    
	 // allocate mem for the result on host side
	cudaMallocHost(&h_cost, no_of_nodes * sizeof(int));

	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;

	// printf("Read File\n");

    //-------------------Allocate memory for host variables ----------------------





	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
    
    
    //Copy the Edge List to device Memory
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;

    //Copy the Mask to device memory
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;

	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;

    //Copy the Visited nodes array to device memory
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;

	
	// allocate device memory for result
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);

    //make a bool to check if the execution is over
	cudaMalloc( (void**) &d_over, sizeof(bool));

    // printf("Allocated memory for CPU\n");
	  
  	// setup execution parameters
	grid = dim3( blocksPerLaunch, 1, 1);

	threads = dim3( num_of_threads_per_block, 1, 1);
    printf("Lauched %d kernels, of %d blocks, with %d threads\n", launches, grid.x, threads.x);
 // Graph edges is not alligned with nodes indexes, so this has to be shared
    cudaMemcpyAsync( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice, stream) ;

        //Alocate memory per kernel
    cudaMemcpyAsync( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice,stream) ;
    cudaMemcpyAsync( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice, stream) ;
    cudaMemcpyAsync( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice, stream) ;
    cudaMemcpyAsync( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice, stream) ;
    cudaMemcpyAsync( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice, stream) ; 
  
	return 0;

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
int launcher( int partitions, cudaStream_t stream) 
{
        for (int i = 0; i < launches; i++) {
            int offset = i * chunkSize;
            Kernel<<<grid, threads, 0, stream>>>(d_graph_nodes + offset, d_graph_edges,
                d_graph_mask + offset, d_updating_graph_mask,
                d_graph_visited, d_cost, chunkSize, offset);

		}
	return 0;
}

int cleaner( cudaStream_t stream ) 
{
	// //Store the result into a file
	// FILE *fpo = fopen("strippedbfs1.txt","w");
	// for(int i=0;i<no_of_nodes;i++)
	// 	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	// fclose(fpo);
	// printf("Result stored in strippedbfs1.txt\n");

    cudaMemcpyAsync( h_updating_graph_mask, d_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyDeviceToHost, stream) ;
    cudaMemcpyAsync( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost, stream) ; 
	cudaDeviceSynchronize();

	// cleanup memory
	cudaFreeHost(h_graph_nodes);
	cudaFreeHost(h_graph_edges);
	cudaFreeHost(h_graph_mask);
	cudaFreeHost(h_updating_graph_mask);
	cudaFreeHost(h_graph_visited);
	cudaFreeHost(h_cost);
	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);
	return 0;
}
}
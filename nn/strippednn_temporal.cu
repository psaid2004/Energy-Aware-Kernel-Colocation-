/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <unistd.h>

// #include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#define minnn( a, b )			a > b ? b : a
#define ceilDivnn( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors



typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;



namespace nn
{
  
	static float *distances, *d_distances;
	//Pointers to device memory
	static LatLong *d_locations;
  static float lat, lng;
	static int chunkSize,numRecords,quiet=0,timing=0,platform=0,device=0;

  
  static std::vector<Record> records;
	static std::vector<LatLong> locations;
	static int resultsCount=10;

  static unsigned long threadsPerBlock, blocks;
  static dim3 grid;
  static int launches;
  	



/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng, int offset)
{
  #pragma unroll 1
  for (int i = 0; i <5000000; i++){
	int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x + offset;
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	}
}
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
  FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

/**
* This program finds the k-nearest neighbors
**/

int setup(int partitions, cudaStream_t stream){
	
char filename[100] = "../nn/list64k_4";
lat = 30.0f;
  lng = 90.0f;
  quiet = 0;
  timing = 0;
  platform = -1;
  device = -1;    

  chdir("../nn/");
  numRecords = loadData(filename, records, locations); // 65536 records
  chdir("../Framework/");   

  printf("Loaded %d records\n", numRecords);
  // printf("Loaded %d records\n", numRecords);
  if (resultsCount > numRecords) resultsCount = numRecords;

    //for(i=0;i<numRecords;i++)
    //  printf("%s, %f, %f\n",(records[i].recString),locations[i].lat,locations[i].lng);

  

  int totalBlocks     = 240;
  threadsPerBlock     = DEFAULT_THREADS_PER_BLOCK;     // 256
  int blocksPerLaunch = 240/partitions;
  launches            = totalBlocks / blocksPerLaunch;  

  numRecords          = totalBlocks * threadsPerBlock;  
  chunkSize           = blocksPerLaunch * threadsPerBlock; 

  grid = dim3(blocksPerLaunch, 1);

	printf("Launched %d kernels with: %d blocks, Threads: %d\n", launches, blocksPerLaunch, threadsPerBlock); //65536 threads, but can be changed

  	
	/**
	* Allocate memory on host and device
	*/
  cudaMallocHost(&distances, numRecords*sizeof(float));
	cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);

  cudaMemcpyAsync(d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice, stream);

  return 0;
}

int launcher(int partitions, cudaStream_t stream)
{
    for(int i=0; i < launches; i++){
        int offset = i *chunkSize;
        euclid<<< grid, threadsPerBlock,0, stream >>>(d_locations+offset,d_distances+offset,numRecords,lat,lng, offset);
        }

    return 0;
}



int cleaner(cudaStream_t stream){
    cudaMemcpyAsync(distances, d_distances, numRecords *sizeof(float), cudaMemcpyDeviceToHost, stream);  
cudaDeviceSynchronize();
    
    findLowest(records,distances,numRecords,resultsCount);

  cudaFreeHost(distances);
  //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);
  return 0;

}

}
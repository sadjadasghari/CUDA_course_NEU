#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TIMER_CREATE(t)               \
cudaEvent_t t##_start, t##_end;     \
cudaEventCreate(&t##_start);        \
cudaEventCreate(&t##_end);               


#define TIMER_START(t)                \
cudaEventRecord(t##_start);         \
cudaEventSynchronize(t##_start);    \


#define TIMER_END(t)                             \
cudaEventRecord(t##_end);                      \
cudaEventSynchronize(t##_end);                 \
cudaEventElapsedTime(&t, t##_start, t##_end);  \
cudaEventDestroy(t##_start);                   \
cudaEventDestroy(t##_end);     

#define TILE_SIZE 16
#define CUDA_TIMING

unsigned char *input_gpu;
unsigned char *output_gpu;
unsigned int *histogram;
unsigned int *cumhistogram; 
unsigned int *SK;

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		exit(-1);
	}
	#endif
	return result;
}

// Add GPU kernel and functions

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel(unsigned char *input,     //generates histogram
	unsigned int *histogram){
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int location = y*TILE_SIZE*gridDim.x+x;
	int myItem = input[location];
	int myBin = myItem % 256;
	atomicAdd(&(histogram[myBin]),1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel3(double alpha, //generates Mapped Function
	unsigned int *cumhistogram,
	unsigned int *SK){    
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int ID = y*TILE_SIZE*gridDim.x+x;
	SK[ID]=cumhistogram[ID]*alpha;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernel4(unsigned char *input,     //generates Output Image
	unsigned int *SK,
	unsigned char *output_gpu){
	int x = blockIdx.x*TILE_SIZE+threadIdx.x;
	int y = blockIdx.y*TILE_SIZE+threadIdx.y;
	int location = y*TILE_SIZE*gridDim.x+x;
	output_gpu[location] = (unsigned char)SK[input[location]]; 
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void histogram_gpu(unsigned char *data, 
	unsigned int height, 
	unsigned int width){

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	

	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu     , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu    , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&histogram     , 256*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&cumhistogram  , 256*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&SK            , 256*sizeof(unsigned int)));

	checkCuda(cudaMemset(output_gpu ,   0, size*sizeof(unsigned char)));
	checkCuda(cudaMemset(histogram,     0, 256*sizeof(unsigned int)));
	checkCuda(cudaMemset(cumhistogram,     0, 256*sizeof(unsigned int)));
	checkCuda(cudaMemset(SK,            0, 256*sizeof(unsigned int)));
    // Copy data to GPU
	checkCuda(cudaMemcpy(input_gpu,
		data,
		size*sizeof(char),
		cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

	dim3 dimGrid(gridXSize, gridYSize);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	dim3 dimGrid2(1, 1);
	dim3 dimBlock2(TILE_SIZE, TILE_SIZE);

	dim3 dimGrid3(1, 1);
	dim3 dimBlock3(TILE_SIZE, TILE_SIZE);

    // Kernel Call
	#if defined(CUDA_TIMING)
	float Ktime;
	TIMER_CREATE(Ktime);
	TIMER_START(Ktime);
	#endif
	kernel<<<dimGrid, dimBlock>>>(input_gpu, 
		histogram);
    //double *alpha2 = new(double);
	double alpha2 = 255/(double)size;
	checkCuda(cudaDeviceSynchronize());

    // generate cumhistogram
	unsigned int *cum_histogram;
	cum_histogram= new(unsigned int[256]);
	unsigned int *histogram2;
	histogram2 = new(unsigned int[256]);
	checkCuda(cudaMemcpy(histogram2,
						 histogram,
						 256*sizeof(unsigned int),
						 cudaMemcpyDeviceToHost));
	cum_histogram[0]=histogram2[0];
	for (int i=1;i<256;i++)
	{
		cum_histogram[i]=cum_histogram[i-1]+histogram2[i];
	}

	checkCuda(cudaMemcpy(cumhistogram,
		cum_histogram,
		256*sizeof(unsigned int),
		cudaMemcpyHostToDevice));
    
	//checkCuda(cudaMemset(cumhistogram,cum_histogram, 256*sizeof(unsigned int)));
	kernel3<<<dimGrid2, dimBlock2>>>(alpha2,
									 cumhistogram,
									 SK);

	checkCuda(cudaDeviceSynchronize());

	kernel4<<<dimGrid, dimBlock>>>(input_gpu,
								   SK,
								   output_gpu);

	#if defined(CUDA_TIMING)
	TIMER_END(Ktime);
	printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
		output_gpu, 
		size*sizeof(unsigned char), 
		cudaMemcpyDeviceToHost));

        // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
	checkCuda(cudaFree(histogram));
}
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
void histogram_gpu_warmup(unsigned char *data,
	unsigned int height, 
	unsigned int width){

	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	int XSize = gridXSize*TILE_SIZE;
	int YSize = gridYSize*TILE_SIZE;
	

	// Both are the same size (CPU/GPU).
	int size = XSize*YSize;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu     , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu    , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&histogram     , 256*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&cumhistogram  , 256*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&SK            , 256*sizeof(unsigned int)));

	checkCuda(cudaMemset(output_gpu ,   0, size*sizeof(unsigned char)));
	checkCuda(cudaMemset(histogram,     0, 256*sizeof(unsigned int)));
	checkCuda(cudaMemset(cumhistogram,     0, 256*sizeof(unsigned int)));
	checkCuda(cudaMemset(SK,            0, 256*sizeof(unsigned int)));
    // Copy data to GPU
	checkCuda(cudaMemcpy(input_gpu,
		data,
		size*sizeof(char),
		cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());

    // Execute algorithm

	dim3 dimGrid(gridXSize, gridYSize);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	dim3 dimGrid2(1, 1);
	dim3 dimBlock2(TILE_SIZE, TILE_SIZE);

	dim3 dimGrid3(1, 1);
	dim3 dimBlock3(TILE_SIZE, TILE_SIZE);

    // Kernel Call
	#if defined(CUDA_TIMING)
	float Ktime;
	TIMER_CREATE(Ktime);
	TIMER_START(Ktime);
	#endif
	kernel<<<dimGrid, dimBlock>>>(input_gpu, 
		histogram);
    //double *alpha2 = new(double);
	double alpha2 = 255/(double)size;
	checkCuda(cudaDeviceSynchronize());

    // generate cumhistogram
	unsigned int *cum_histogram;
	cum_histogram= new(unsigned int[256]);
	unsigned int *histogram2;
	histogram2 = new(unsigned int[256]);
	checkCuda(cudaMemcpy(histogram2,
						 histogram,
						 256*sizeof(unsigned int),
						 cudaMemcpyDeviceToHost));
	cum_histogram[0]=histogram2[0];
	for (int i=1;i<256;i++)
	{
		cum_histogram[i]=cum_histogram[i-1]+histogram2[i];
	}

	checkCuda(cudaMemcpy(cumhistogram,
		cum_histogram,
		256*sizeof(unsigned int),
		cudaMemcpyHostToDevice));
    
	//checkCuda(cudaMemset(cumhistogram,cum_histogram, 256*sizeof(unsigned int)));
	kernel3<<<dimGrid2, dimBlock2>>>(alpha2,
									 cumhistogram,
									 SK);

	checkCuda(cudaDeviceSynchronize());

	kernel4<<<dimGrid, dimBlock>>>(input_gpu,
								   SK,
								   output_gpu);

	#if defined(CUDA_TIMING)
	TIMER_END(Ktime);
	printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif

	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
		output_gpu, 
		size*sizeof(unsigned char), 
		cudaMemcpyDeviceToHost));

        // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
	checkCuda(cudaFree(histogram));

}


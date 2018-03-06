#include <stdio.h>
#include <stdlib.h>

#define block_size   32
#define vector_size  100

__global__ void add( int *a, int *b, int *c ) {
        int tid = (blockIdx.x*blockDim.x) + threadIdx.x;    // this thread handles the data at its thread id

        if (tid < vector_size){
                c[tid] = a[tid] + b[tid];                   // add vectors together                
        }
}

int main( void ) { 

        // Set device that we will use for our cuda code
        // It will be either 0 or 1
        cudaSetDevice(0);
        
	// Time Variables
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
        
	// Input Arrays and variables
        int *a        = new int [vector_size]; 
        int *b        = new int [vector_size]; 
        int *c_cpu    = new int [vector_size]; 
        int *c_gpu    = new int [vector_size];
        
	// Pointers in GPU memory
        int *dev_a;
        int *dev_b;
        int *dev_c;
        
        // fill the arrays 'a' and 'b' on the CPU
        for (int i = 0; i < vector_size; i++) {
                a[i] = rand()%10;
                b[i] = rand()%10;
        }

        //
        // CPU Calculation
        //////////////////
        
	printf("Running sequential job.\n");
	cudaEventRecord(start,0);
        
        // Calculate C in the CPU
        for (int i = 0; i < vector_size; i++) {
                c_cpu[i] = a[i] + b[i];
        }
        
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("\tSequential Job Time: %.2f ms\n", time);

        // allocate the memory on the GPU
        cudaMalloc( (void**)&dev_a,  vector_size * sizeof(int) );
        cudaMalloc( (void**)&dev_b,  vector_size * sizeof(int) );
        cudaMalloc( (void**)&dev_c,  vector_size * sizeof(int) );

        // copy the arrays 'a' and 'b' to the GPU
        cudaMemcpy( dev_a, a, vector_size * sizeof(int),
                              cudaMemcpyHostToDevice );
        cudaMemcpy( dev_b, b, vector_size * sizeof(int),
                              cudaMemcpyHostToDevice );

        //
        // GPU Calculation
        ////////////////////////
        
        printf("Running parallel job.\n");
	
        int grid_size = (vector_size-1)/block_size;
        grid_size++;
        
	cudaEventRecord(start,0);
        add<<<grid_size,block_size>>>( dev_a, dev_b, dev_c);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	printf("\tParallel Job Time: %.2f ms\n", time);

        // copy the array 'c' back from the GPU to the CPU
        cudaMemcpy( c_gpu, dev_c, vector_size * sizeof(int), 
                                cudaMemcpyDeviceToHost );

        // compare the results
        int error = 0;
        for (int i = 0; i < vector_size; i++) {
                if (c_cpu[i] != c_gpu[i]){
                        error = 1;
                        printf( "Error starting element %d, %d != %d\n", i, c_gpu[i], c_cpu[i] );    
                }
		if (error) break; 
        }
        
        if (error == 0){
                printf ("Correct result. No errors were found.\n");
        }
        
        // free the memory allocated on the GPU
        cudaFree( dev_a );
        cudaFree( dev_b );
        cudaFree( dev_c );

	free(a);
	free(b);
	free(c_cpu);
	free(c_gpu);

        return 0;
}


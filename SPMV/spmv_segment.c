#include "genresult.cuh"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void segmented_scan(
					const int lane,
					const int *rows,
					float * vals
					){
		if (lane >=1 && rows[threadIdx.x] == rows[threadIdx.x-1])
			vals[threadIdx.x] += vals[threadIdx.x-1];
		if (lane >=2 && rows[threadIdx.x] == rows[threadIdx.x-2])
			vals[threadIdx.x] += vals[threadIdx.x-2];
		if (lane >=4 && rows[threadIdx.x] == rows[threadIdx.x-4])
			vals[threadIdx.x] += vals[threadIdx.x-4];
		if (lane >=8 && rows[threadIdx.x] == rows[threadIdx.x-8])
			vals[threadIdx.x] += vals[threadIdx.x-8];
		if (lane >=16 && rows[threadIdx.x] == rows[threadIdx.x-16])
			vals[threadIdx.x] += vals[threadIdx.x-16];
		
}

__global__ void putProduct_kernel(
					const int nnz,
					const int *coord_row,
					const int *coord_col,
					const float *A,
					const float *x,
					float *y){
    
		int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
		int thread_num = blockDim.x * gridDim.x;
		int iter = nnz % thread_num ? nnz / thread_num +1 : nnz / thread_num;
		int i;
		
		/* Shared memory declarations */
		__shared__ int rowIndices[sizeof(int)*256];
		__shared__ float multiplications[sizeof(float)*256];
		
		for(i = 0; i < iter; i++){
			/* data id is the id of data */
			int dataid = thread_id + i*thread_num;
			
			if (dataid < nnz){
				float data = A[dataid];
				int row = coord_row[dataid];
				int col = coord_col[dataid];
				float temp = data * x[col];
				
				/*update shared memory -- multiplications && rowIndices */
				multiplications[threadIdx.x] = temp;
				rowIndices[threadIdx.x] = row;
				
				/*call segmented_scan*/
				segmented_scan(threadIdx.x % 32,rowIndices,multiplications);
				
				/*last one in warp || last thread */
				if (threadIdx.x % 32 == 31 || dataid == nnz-1){
					row = rowIndices[threadIdx.x];
					temp = multiplications[threadIdx.x];
					atomicAdd(&y[row], temp);
				}
				/*if row index is different from the next thread */
				else if (rowIndices[threadIdx.x] != rowIndices[threadIdx.x+1]){
					row = rowIndices[threadIdx.x];
					temp = multiplications[threadIdx.x];
					atomicAdd(&y[row], temp);
				}
			}
		}
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    
	
	/*Allocate things...*/
	/**/
	
	cudaError_t err = cudaSuccess;
	int size = mat -> nz;
	printf("Start Device Memory Allocation \n");
	
	/*Device Allocation -- coord_row*/
	int * coord_row =NULL;
	err = cudaMalloc((void **)&coord_row, size*sizeof(int));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device coord_row (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/*Device Allocation -- coord_col*/
	int * coord_col =NULL;
	err = cudaMalloc((void **)&coord_col, size*sizeof(int));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device coord_col (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
	/*Device Allocation -- value of Matrix A*/
	float * d_A =NULL;
	err = cudaMalloc((void **)&d_A, size*sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/*Device Allocation -- d_x Vector*/
	float * d_x =NULL;
	err = cudaMalloc((void **)&d_x, vec->M*sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_x Vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/*Device Allocation -- d_y Vector*/
	float * d_y =NULL;
	err = cudaMalloc((void **)&d_y, mat->M*sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_y Result (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	printf("Start Copying contents \n");
	
	/*Device Memcpy */
	err = cudaMemcpy(coord_row, mat->rIndex, size*sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy rindex from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaMemcpy(coord_col, mat->cIndex, size*sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy cindex from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaMemcpy(d_A, mat->val, size*sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy val from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaMemcpy(d_x, vec->val, vec->M*sizeof(float), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	float *init = (float *)calloc(mat->M,sizeof(float));

    err = cudaMemcpy(d_y, init, mat->M*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy Vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	/**/
	/**/
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernel(s)*/
	/**/
	putProduct_kernel<<<blockNum,blockSize>>>(size, coord_row,coord_col,d_A,d_x,d_y);
	err = cudaGetLastError();
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch spmvAtomic kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	/* Write back from Device*/
	printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(res->val, d_y, mat->M * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy d_y from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	/**/

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate, please*/
	
	/**/
	err = cudaFree(coord_row);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device coord_row (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaFree(coord_col);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device coord_col (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaFree(d_A);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_A(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaFree(d_x);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_x (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaFree(d_y);
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_y (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	free(init);
	
	err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	/**/
}

#include "genresult.cuh"
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>



typedef struct{
	int row;
	int num;
}rowstruct;

typedef struct ohyah{
	int index;
	struct ohyah * next;
}list;

int cmpfunc (const void * a, const void * b)
{
	rowstruct * ap = (rowstruct *)a;
	rowstruct * bp = (rowstruct *)b;
   return ( ap-> num - bp->num );
}

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /*change rows according to count to avoid thread Divergence */
	list ** array = (list **) calloc(mat->M, sizeof(list *));
        int * count = (int *)calloc(mat -> M, sizeof(int));
	rowstruct * hello = (rowstruct *) calloc(mat -> M, sizeof(rowstruct));
        float * tempval = (float *)calloc(mat->nz, sizeof(float));
        int * temprIndex =(int *) calloc(mat->nz, sizeof(int));
        int * tempcIndex =(int *)calloc(mat->nz, sizeof(int));
        int i;
        
        /*Count the numbers of appearce of each row*/
        for(i=0; i< mat-> nz; i++){
                count[mat->rIndex[i]]++;
		list * prev = array[mat->rIndex[i]];
		/*fill in linked list*/
		if (prev == NULL){
			list * add = (list * )calloc(1, sizeof(list));
			add->index = i;
			add->next = NULL;
			array[mat->rIndex[i]] = add;
		}else{
			while(prev->next != NULL){
				prev = prev->next;
			}
			list * add = (list * )calloc(1, sizeof(list));
			add->index = i;
			add->next = NULL;
			prev->next = add;
		}
		
        }

        for(i=0;i< mat->M;i++){
		hello[i].row = i;
		hello[i].num = count[i];
	}

	qsort (hello, mat->M, sizeof(rowstruct), cmpfunc);
	
	int countrow = 0;
	for(i=0; i<mat->M; i++){
		int rowindex = hello[i].row;
		list * node = array[rowindex];
		while( node != NULL){
			temprIndex[countrow] = mat->rIndex[node->index];
			tempcIndex[countrow] = mat->cIndex[node->index];
			tempval[countrow] = mat->val[node->index];
			countrow++;
			node = node->next;
		}
	}


        memcpy(mat ->rIndex , temprIndex, mat->nz * sizeof(int));
        memcpy(mat ->cIndex , tempcIndex, mat->nz * sizeof(int));
        memcpy(mat -> val, tempval, mat->nz * sizeof(float));

        free(tempval);
        free(tempcIndex);
        free(temprIndex);
        free(count);

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
    /*Your own magic here*/
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
    printf("Your Own Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

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

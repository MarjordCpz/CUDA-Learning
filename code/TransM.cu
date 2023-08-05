#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define M 500
#define N 400
#define BLOCK_SIZE 16
#define CEIL(X,Y) ((X+Y-1)/Y)

__managed__ int Matrix_in[M][N];
__managed__ int Matrix_out_gpu1[N][M];
__managed__ int Matrix_out_gpu2	[N][M];
__managed__ int Matrix_out_cpu[N][M];

void TranposeM_cpu(const int matrix_in[M][N], int matrix_out[N][M]) {
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < M; x++) {
			matrix_out[y][x] = matrix_in[x][y];
		}
	}
}


__global__ void TransposeM_v1(const int matrix_in[M][N], int matrix_out[N][M]) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < M && y < N) {
		matrix_out[y][x] = matrix_in[x][y];
	}
}

__global__ void TransposeM_v2(const int matrix_in[M][N], int matrix_out[N][M]) {
	/*
		大致流程为;
		- 将global mem 存到shared mem
		- 从shared mem 中读取数据(不同位置的)，写到global mem 中
	*/
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	__shared__ int temp[BLOCK_SIZE][BLOCK_SIZE + 1];	//为了避免出现bank冲突进行padding操作，详细分析见readme

	if (x < N && y < M) {
		temp[threadIdx.y][threadIdx.x] = matrix_in[y][x];
	}
	__syncthreads();

	int x1 = blockDim.y * blockIdx.y + threadIdx.x;
	int y1 = blockDim.x * blockIdx.x + threadIdx.y;

	if (x1 < M && y1 < N) {
		matrix_out[y1][x1] = temp[threadIdx.x][threadIdx.y];	//反着读
	}

}

int main(void) {

	//init
	for (int y = 0; y < M; y++) {
		for (int x = 0; x < N; x++) {
			Matrix_in[y][x] = rand() % 20;
		}
	}
	//block and grid
	unsigned int grid_size_x = CEIL(N, BLOCK_SIZE);
	unsigned int grid_size_y = CEIL(M, BLOCK_SIZE);
	dim3 grid_size(grid_size_x, grid_size_y);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	cudaEvent_t start1, start2, start3, stop1, stop2, stop3;
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	cudaEventCreate(&stop3);

	cudaEventRecord(start1);
	for (int i = 0; i < 20; i++) {
		TransposeM_v1 <<<grid_size,block_size>>> (Matrix_in, Matrix_out_gpu1);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);
	

	cudaEventRecord(start2);
	for (int i = 0; i < 20; i++) {
		TransposeM_v2 <<<grid_size, block_size>>> (Matrix_in, Matrix_out_gpu2);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);

	cudaEventRecord(start3);
	for (int i = 0; i < 20; i++) {
		TranposeM_cpu(Matrix_in, Matrix_out_cpu);
	}
	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);

	float time1, time2, time3;
	cudaEventElapsedTime(&time1, start1, stop1);
	cudaEventElapsedTime(&time2, start2, stop2);
	cudaEventElapsedTime(&time3, start3, stop3);
	printf("Time_gpu1 is %.5f ms.\n",time1);
	printf("Time_gpu2 is %.5f ms.\n",time2);
	printf("Time_cpu is %.5f ms.\n", time3);
	cudaEventDestroy(start1);
	cudaEventDestroy(start2);
	cudaEventDestroy(start3);
	cudaEventDestroy(stop1);
	cudaEventDestroy(stop2);
	cudaEventDestroy(stop3);

	//test
	bool errors = false;
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < M; x++) {
			if (fabs(Matrix_out_cpu[y][x] - Matrix_out_gpu2[y][x]) > 1.0e-15) {
				errors = true;
			}
		}
	}
	printf("Result is %s.\n", errors ? "error" : "pass");

	return 0;
}

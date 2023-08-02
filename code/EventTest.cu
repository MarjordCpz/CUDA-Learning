/*基于矩阵乘法的时间计算*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"
#include "error.cuh"


// M*K * K*N == M*N
#define M 1000
#define N 1000
#define K 1000

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 32

#define Ceil(x, y) ((x+y-1)/y)

//naive version
__global__ void MatrixMult(int* a, int* b, int* c, const int m, const int n, const int k) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < n && y < m) {
		int temp = 0;
		for (int step = 0; step < k; step++) {
			temp += a[y * k + step] * b[n * step + x];
		}
		c[y * n + x] = temp;
	}
}

void cpu_mm(int* a, int* b, int* c, const int m, const int n, const int k) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int temp = 0;
			for (int step = 0; step < k; step++) {
				temp += a[i * k + step] * b[step * n + j];
			}
			c[i * n + j] = temp;
		}
	}
}

int main(void) {
	int matrix1_size_row = M;
	int matrix1_size_col = K;
	int matrix2_size_row = K;
	int matrix2_size_col = N;
	int matrix3_size_row = M;
	int matrix3_size_col = N;
	int memsize1 = sizeof(int) * matrix1_size_row * matrix1_size_col;
	int memsize2 = sizeof(int) * matrix2_size_row * matrix2_size_col;
	int memsize3 = sizeof(int) * matrix3_size_row * matrix3_size_col;

	//cpu 
	int* h_a, * h_b, * h_c, * h_result;
	cudaMallocHost((void**)&h_a, memsize1);
	cudaMallocHost((void**)&h_b, memsize2);
	cudaMallocHost((void**)&h_c, memsize3);
	cudaMallocHost((void**)&h_result, memsize3);

	//initialize
	for (int i = 0; i < matrix1_size_row; i++) {
		for (int j = 0; j < matrix1_size_col; j++) {
			h_a[i * matrix1_size_col + j] = rand() % 1024;
		}
	}
	for (int i = 0; i < matrix2_size_row; i++) {
		for (int j = 0; j < matrix2_size_col; j++) {
			h_b[i * matrix2_size_col + j] = rand() % 1024;
		}
	}




	//gpu
	int* d_a, * d_b, * d_c;
	cudaMalloc((void**)&d_a, memsize1);
	cudaMalloc((void**)&d_b, memsize2);
	cudaMalloc((void**)&d_c, memsize3);

	// time test
	cudaEvent_t start, stop_cpu, stop_gpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_cpu);
	cudaEventCreate(&stop_gpu);

	cudaEventRecord(start);
	//initialize
	cudaMemcpy(d_a, h_a, memsize1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, memsize2, cudaMemcpyHostToDevice);

	//grid and block
	unsigned int grid_size_row = Ceil(M, BLOCK_SIZE_Y);
	unsigned int grid_size_col = Ceil(N, BLOCK_SIZE_X);
	dim3 grid_size(grid_size_col, grid_size_row, 1);
	dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);


	//calculate
	MatrixMult <<<grid_size, block_size>>> (d_a, d_b, d_c, M, N, K);
	cudaMemcpy(h_result, d_c, memsize3, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop_gpu);
	cudaEventSynchronize(stop_gpu);

	cpu_mm(h_a, h_b, h_c, M, N, K);
	cudaEventRecord(stop_cpu);
	cudaEventSynchronize(stop_cpu);

	// time comparing
	float time_cpu, time_gpu;
	cudaEventElapsedTime(&time_gpu, start, stop_gpu);
	cudaEventElapsedTime(&time_cpu, start, stop_cpu);

	printf("GPU time is %.7f ms\n", time_gpu);
	printf("CPU time is %.7f ms\n", time_cpu);


	cudaEventDestroy(start);
	cudaEventDestroy(stop_cpu);
	cudaEventDestroy(stop_gpu);


	//test
	bool error = false;
	
	int cnt = 0;
	for (int y = 0; y < M; y++) {
		for (int x = 0; x < N; x++) {
			if (fabs(h_result[y * N + x] - h_c[y * N + x]) > 1.0e-10) {
				error = true;
				cnt++;
			}
		}
	}
	//printf("cnt is %d\n", cnt);
	printf("Result is %s\n", error ? "error" : "pass");

	//free
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_result);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);



	return 0;
}
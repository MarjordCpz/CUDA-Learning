/*
待查询&解决的问题: 
内存分配函数  cudaMallocHost( )。
如何优化矩阵相乘的函数，需要深入了解资源优化，访存设计等 https://zhuanlan.zhihu.com/p/410278370。
*/
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"
#include <stdlib.h>

// M*K * K*N == M*N
#define M 500
#define N 400
#define K 300

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 32

#define Ceil(x, y) ((x+y-1)/y)

//naive version
__global__ void MatrixMult(int* a, int* b, int* c, const int m, const int n, const int k) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < n && y < m) {
		//printf("(x, y) == (%d, %d)\n", x, y);
		int temp = 0;
		for (int step = 0; step < k; step++) {
			temp += a[y * k + step] * b[n * step + x];
		}
		c[y * n + x] = temp;
	}
	//else {
	//	printf(" None: (x, y) == (%d, %d)\n", x, y);
	//}
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

	//initialize
	cudaMemcpy(d_a, h_a, memsize1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, memsize2, cudaMemcpyHostToDevice);

	//grid and block
	unsigned int grid_size_row = Ceil(M, BLOCK_SIZE_Y);
	unsigned int grid_size_col = Ceil(N, BLOCK_SIZE_X);
	dim3 grid_size(grid_size_col, grid_size_row, 1);
	dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

	//error
	//dim3 grid_size(grid_size_row, grid_size_col, 1);
	//dim3 block_size(BLOCK_SIZE_Y, BLOCK_SIZE_X, 1);

	printf("(%d, %d)\n", grid_size.x, grid_size.y);
	printf("(%d, %d)\n", block_size.x, block_size.y);

	//calculate
	MatrixMult<<<grid_size, block_size>>> (d_a, d_b, d_c, M, N, K);
	cpu_mm(h_a, h_b, h_c, M, N, K);

	//test
	bool error = false;
	cudaMemcpy(h_result, d_c, memsize3, cudaMemcpyDeviceToHost);
	int cnt = 0;
	for (int y = 0; y < M; y++) {
		for (int x = 0; x < N; x++) {
			if (fabs(h_result[y * N + x] - h_c[y * N + x]) > 1.0e-10) {
				error = true;
				cnt++;
			}
		}
	}
	printf("cnt is %d\n", cnt);
	printf("Result is %s\n", error ? "error": "pass");

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
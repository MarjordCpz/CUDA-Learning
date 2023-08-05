#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M 500
#define N 400
#define K 300
// a (M*K) * b (K*N) -> c (M*N)

#define BLOCK_SIZE 32


#define CEIL(X, Y) ((X + Y - 1) / Y)
/*
__managed__ 
 - 可以和 __device__ 联合使用
 - 可以被主机和设备引用，主机或者设备函数可以获取其地址或者读写其值
 - 生命周期为整个应用期间

 - 统一内存消除了通过 cudaMemcpy*() 例程进行显式数据移动的需要，而不会因将所有数据放入零拷贝内存而导致性能损失。
 - 当然，数据移动仍然会发生，因此程序的运行时间通常不会减少；相反，统一内存可以编写更简单、更易于维护的代码。


*/
__managed__ int a[M * K];
__managed__ int b[K * N];
__managed__ int c_gpu1[M * N];
__managed__ int c_gpu2[M * N];
__managed__ int c_cpu[M * N];


__global__ void MM_V2(const int* a, const int* b, int* c, const int m, const int n, const int k) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;

	__shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

	int temp = 0;
	int idx;
	for (int step = 0; step <= k / BLOCK_SIZE; step++) {
		// sub_a
		int step_x = step * BLOCK_SIZE + threadIdx.x;
		int step_y = y;
		idx = step_y * k + step_x;
		if (step_x >= k || step_y >= m) {
			sub_a[threadIdx.y][threadIdx.x] = 0;
		}
		else {
			sub_a[threadIdx.y][threadIdx.x] = a[idx];
		}
		// sub_b
		step_x = x;
		step_y = step * BLOCK_SIZE + threadIdx.y;
		idx = step_y * n + step_x;
		if (step_x >= n || step_y >= k) {
			sub_b[threadIdx.y][threadIdx.x] = 0;
		}
		else {
			sub_b[threadIdx.y][threadIdx.x] = b[idx];
		}
		__syncthreads();
		// multiple
		for (int i = 0; i < BLOCK_SIZE; i++) {
			temp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
		}
		__syncthreads();
	}		
	if (x < n && y < m) {
		c[y * n + x] = temp;
	}
}


__global__ void MM_V1(const int* a, const int* b, int* c, const int m, const int n, const int k) {
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



void MM_cpu(const int* a, const int* b, int* c, const int m, const int n, const int k) {
	for (int y = 0; y < m; y++) {
		for (int x = 0; x < n; x++) {
			int temp = 0;

			for (int step = 0; step < k; step++) {
				temp += a[step + y * k] + b[x + step * n];
			}
			c[y * n + x] = temp;
		}
	}
}


int main(void) {

	//initialize
	for (int y = 0; y < M; y++) {
		for (int x = 0; x < K; x++) {
			a[y * K + x] = rand() % 20;

		}
	}
	for (int y = 0; y < K; y++) {
		for (int x = 0; x < N; x++) {
			b[y * N + x] = rand() % 20;

		}
	}

	//time
	cudaEvent_t start1, start2, start3, stop1, stop2, stop3;
	cudaEventCreate(&start1);
	cudaEventCreate(&start2);
	cudaEventCreate(&start3);
	cudaEventCreate(&stop1);
	cudaEventCreate(&stop2);
	cudaEventCreate(&stop3);

	//grid and block
	unsigned int grid_size_x = CEIL(N, BLOCK_SIZE);
	unsigned int grid_size_y = CEIL(M, BLOCK_SIZE);

	dim3 grid_size(grid_size_x, grid_size_y);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

	//calculate
	cudaEventRecord(start1);
	MM_V1 <<< grid_size,block_size >>> (a, b, c_gpu1, M, N, K);
	cudaDeviceSynchronize();
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

	cudaEventRecord(start2);
	MM_V2 <<< grid_size,block_size >>> (a, b, c_gpu2, M, N, K);
	cudaDeviceSynchronize();
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);

	cudaEventRecord(start3);
	MM_cpu(a, b, c_cpu, M, N, K);
	cudaDeviceSynchronize();
	cudaEventRecord(stop3);
	cudaEventSynchronize(stop3);


	float time1, time2, time3, time_test;
	cudaEventElapsedTime(&time1, start1, stop1);
	cudaEventElapsedTime(&time2, start2, stop2);
	cudaEventElapsedTime(&time3, start2, stop3);
	//cudaEventElapsedTime(&time_test, start3, stop3);
	printf("Time_gpu1 is %.5f ms.\n",time1);
	printf("Time_gpu2 is %.5f ms.\n", time2);
	printf("Time_cpu is %.5f ms.\n",time3);
	//printf("Time_test is %.5f ms.\n", time_test);

	cudaEventDestroy(start1);
	cudaEventDestroy(start2);
	cudaEventDestroy(start3);
	cudaEventDestroy(stop1);
	cudaEventDestroy(stop2);
	cudaEventDestroy(stop3);


	//test
	bool error = false;
	for (int y = 0; y < M; y++) {
		for (int x = 0; x < N; x++) {
			if (fabs(c_gpu2[y * N + x] - c_gpu1[y * N + x]) > 1.0e-15) {
				error = true;
			}
		}
	}
	printf("Result is %s. \n", error ? "error" : "pass");


	return 0;
}

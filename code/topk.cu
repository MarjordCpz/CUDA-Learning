#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1000000
#define BLOCK_SIZE 256
#define GRID_SIZE 64
#define TOPK 20

__managed__ int source[N];
__managed__ int gpu_result[TOPK];
__managed__ int _1_pass_result[TOPK * GRID_SIZE];

__device__ __host__ void insert_value(int* array, int k, int data) {
	for (int i = 0; i < k; i++) {
		if (data == array[i]) {
			return;
		}
	}
	if (data < array[k - 1]) {
		return;
	}
	for (int i = k - 2; i >= 0; i--) {
		if (data > array[i]) {
			array[i + 1] = array[i];
		}
		else {
			array[i + 1] = data;
			return;
		}
	}
	array[0] = data;
}

__global__ void topk_gpu(const int* in, int* out, int length) {
	__shared__ int Smem[TOPK * BLOCK_SIZE];
	int top_array[TOPK];

	for (int i = 0; i < TOPK; i++) {
		top_array[i] = INT_MIN;
	}

	for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx+=blockDim.x*gridDim.x) {
		insert_value(top_array, TOPK, in[idx]);
	}
	for (int m = 0; m < TOPK; m++) {
		Smem[threadIdx.x * TOPK + m] = top_array[m];
	}
	__syncthreads();
	for (int step = BLOCK_SIZE / 2; step >= 1; step /= 2) {
		if (threadIdx.x < step) {
			for (int m = 0; m < TOPK; m++) {
				insert_value(top_array, TOPK, Smem[TOPK * (step + threadIdx.x) + m]);
			}
		}
		__syncthreads();
		if (threadIdx.x < step) {
			for (int m = 0; m < TOPK; m++) {
				Smem[TOPK * threadIdx.x + m] = top_array[m];
			}
		}
		__syncthreads();
	}
	if (blockDim.x * blockIdx.x < length) {
		if (threadIdx.x == 0) {
			for (int i = 0; i < TOPK; i++) {
				out[TOPK * blockIdx.x + i] = Smem[i];
			}
		}
	}
	__syncthreads();
}

void cpu_topk(int* in, int* out, int length, int k)
{
	for (int i = 0; i < length; i++)
	{
		insert_value(out, k, in[i]);
	}
}

int main()
{
	printf("Init source data...........\n");
	for (int i = 0; i < N; i++)
	{
		source[i] = rand();
	}

	printf("Complete init source data.....\n");
	cudaEvent_t start, stop_gpu, stop_cpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_gpu);
	cudaEventCreate(&stop_cpu);

	cudaEventRecord(start);
	cudaEventSynchronize(start);
	printf("GPU Run **************\n");
	for (int i = 0; i < 20; i++)
	{
		topk_gpu<<<GRID_SIZE, BLOCK_SIZE >>> (source, _1_pass_result, N);

		topk_gpu<<<1, BLOCK_SIZE >>> (_1_pass_result, gpu_result, TOPK * GRID_SIZE);

		cudaDeviceSynchronize();
	}
	printf("GPU Complete!!!\n");
	cudaEventRecord(stop_gpu);
	cudaEventSynchronize(stop_gpu);

	int cpu_result[TOPK] = { 0 };
	printf("CPU RUN***************\n");
	cpu_topk(source, cpu_result, N, TOPK);
	cudaEventRecord(stop_cpu);
	cudaEventSynchronize(stop_cpu);
	printf("CPU Complete!!!!!");

	float time_cpu, time_gpu;
	cudaEventElapsedTime(&time_gpu, start, stop_gpu);
	cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);

	bool error = false;
	for (int i = 0; i < TOPK; i++)
	{
		printf("CPU top%d: %d; GPU top%d: %d;\n", i + 1, cpu_result[i], i + 1, gpu_result[i]);
		if (fabs(gpu_result[i] - cpu_result[i]) > 0)
		{
			error = true;
		}
	}
	printf("Result: %s\n", (error ? "Error" : "Pass"));
	printf("CPU time: %.2f; GPU time: %.2f\n", time_cpu, (time_gpu / 20.0));
	return 0;
}

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 256
#define GRID_SIZE 32
#define N 1000000

__managed__ int Vec_in[N];
__managed__ int Result[1] = { 0 };


__global__ void Reduce_sum(const int *in, int *out, int count) {
	__shared__ int Smem[BLOCK_SIZE];
	int shared_temp = 0;
	for (int index = threadIdx.x + blockDim.x * blockIdx.x; index < count; index += gridDim.x * blockDim.x) {
		shared_temp += in[index];
	}
	Smem[threadIdx.x] = shared_temp;		//多个blocks的一组threads
	__syncthreads();

	int temp = 0;
	for (int step = blockDim.x / 2; step >= 1; step/=2) {
		//先读出来，再写回去
		if (threadIdx.x < step) {
			temp = Smem[threadIdx.x] + Smem[threadIdx.x + step];
		}
		__syncthreads();	//在分支外使用
		if (threadIdx.x < step) {
			Smem[threadIdx.x] = temp;
		}
		__syncthreads();
	}
	//将每一个block中的首位数据使用原子操作相加
	if (blockDim.x * blockIdx.x < count) {
		if (threadIdx.x == 0) {
			atomicAdd(out,Smem[0]);
		}
	}
}




int main(void) {
	int cpu_result = 0;

	cudaEvent_t start1, stop1, start2, stop2;

	printf("Init input.\n");
	for (int i = 0; i < N; i++) {
		Vec_in[i] = rand() % 2;
	}
	printf("Init is done\n");

	cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);


	Reduce_sum <<<GRID_SIZE, BLOCK_SIZE>>> (Vec_in, Result, N);
	cudaEventRecord(start1);
	for (int i = 0; i < 20; i++) {
		//Result[0] = 0;
		Reduce_sum <<<GRID_SIZE, BLOCK_SIZE>>> (Vec_in, Result, N);
	}
	cudaDeviceSynchronize();
	cudaEventRecord(stop1);
	cudaEventSynchronize(stop1);

	cudaEventRecord(start2);
	for (int i = 0; i < N; i++) {
		cpu_result += Vec_in[i];
	}
	cudaEventRecord(stop2);
	cudaEventSynchronize(stop2);

	float time1, time2;
	cudaEventElapsedTime(&time1, start1, stop1);
	cudaEventElapsedTime(&time2, start2, stop2);
	printf("Time_gpu is %.5f ms.\nTime_cpu is %.5f ms.\n", time1 / 20.0, time2);

	bool errors = false;
	if (fabs(cpu_result - Result[0]/21) > 1.0e-15) {
		errors = true;
	}
	printf("The gpu result is %d.\nThe cpu result is %d.\n", Result[0]/21, cpu_result);
	printf("%s\n", errors ? "Has error!" : "Pass!");

	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);
	cudaEventDestroy(start2);
	cudaEventDestroy(stop2);

	return 0;
}

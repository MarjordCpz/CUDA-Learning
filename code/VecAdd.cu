#include<stdio.h>
#include<math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "malloc.h"

const double a = 1.2;
const double b = 2.3;
//const double c = 3.5;
const int N = 100;
const int M = sizeof(double) * N;


__global__ void VecAdd(const double* d_x, const double* d_y, double* d_z, const int N) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < N) {
		d_z[index] = d_x[index] + d_y[index];
	}
}
void cpu_add(const double* h_x, const double* h_y, double* h_z, const int N) {
	for (int i = 0; i < N; i++) {
		h_z[i] = h_x[i] + h_y[i];
	}
}

__global__ void check_gpu(const double* d_z, const double* gpu_result, const int N, bool test) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N) {
		if (fabs(d_z[index] - gpu_result[index]) > 1.0e-15) {
			printf("Error!\n");
			test = 1;
		}
		else {
			printf("all right\n");
		}
	}
	else {
		printf("None\n");
	}
}

int main(void) {

	// cpu mem alloc
	double* h_x = (double*)malloc(M);
	double* h_y = (double*)malloc(M);
	double* h_z = (double*)malloc(M);
	double* cpu_result = (double*)malloc(M);
	bool test = false;

	for (int i = 0; i < N; i++) {
		h_x[i] = a;
		h_y[i] = b;
	}

	// gpu mem alloc
	double * d_x, * d_y, * d_z, *gpu_result;
	cudaMalloc((void**)&d_x, M);
	cudaMalloc((void**)&d_y, M);
	cudaMalloc((void**)&d_z, M);
	cudaMalloc((void**)&gpu_result, M);

	// mem copy
	cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

	//grid and block
	const int block_size = 128;
	const int grid_size = (N + block_size - 1) / block_size;
	// gpu version
	VecAdd <<<grid_size, block_size>>> (d_x, d_y, d_z, N);
	// cpu version
	cpu_add(h_x, h_y, cpu_result, N);

	cudaMemcpy(gpu_result, cpu_result, M, cudaMemcpyHostToDevice);


	//test
	check_gpu <<<grid_size, block_size>>> (d_z, gpu_result, N, test); 


	printf("Result: %s.\n", test ? "error" : "pass");

	//end
	free(h_x);
	free(h_y);
	free(h_z);
	free(cpu_result);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	return 0;

}
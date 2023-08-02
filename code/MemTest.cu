#include "error.cuh"
#include <stdio.h>
#include "cuda_runtime.h"

__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void my_Kernel() {
	d_y[0] += d_x;
	d_y[1] += d_x;
	printf("d_x is %d, d_y[0] is %d, d_y[1] is %d.\n", d_x, d_y[0], d_y[1]);
}

int main(void) {
	int h_y[2] = { 10, 20 };
	CHECK(cudaMemcpyToSymbol(d_y, h_y, 2 * sizeof(int)));

	my_Kernel <<<1, 1>>> ();
	CHECK(cudaDeviceSynchronize());

	//  warning #20091-D: a __device__ variable "d_y" cannot be directly read in a host function
	//printf("d_x is %d, d_y[0] is %d, d_y[1] is %d.\n", d_x, d_y[0], d_y[1]);

	CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * -1));
	printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);
	return 0;
}
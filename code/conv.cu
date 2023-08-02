/*实现简单的卷积运算，应用Sobel检测的实例*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CEIL(X, Y) ((X+Y-1)/Y) 
#define IMG_HEIGHT 40
#define IMG_WIDTH  50
#define BLOCK_SIZE 16

__global__ void conv_k3(const int* img_in, const int* kernel, int* img_out, const int height, const int width) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = y * width + x;
	//kernel_size is 3
	int x0, x1, x2, x3, x4, x5, x6, x7, x8;		
	if (x > 0 && x < (width - 1) && y>0 && y < (height - 1)) {
		x0 = img_in[(y - 1) * width + x - 1] * kernel[0];
		x1 = img_in[(y - 1) * width + x] * kernel[1];
		x2 = img_in[(y - 1) * width + x + 1] * kernel[2];
		x3 = img_in[y * width + x - 1] * kernel[3];
		x4 = img_in[y * width + x] * kernel[4];
		x5 = img_in[y * width + x + 1] * kernel[5];
		x6 = img_in[(y + 1) * width + x - 1] * kernel[6];
		x7 = img_in[(y + 1) * width + x] * kernel[7];
		x8 = img_in[(y + 1) * width + x + 1] * kernel[8];

		int result = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8;
		img_out[idx] = result;
	}
}

__global__ void sobel_ex(const int* img_in, int* img_out, const int height, const int width) {
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int idx = y * width + x;
	//kernel_size is 3
	int G_x, G_y;
	int x0, x1, x2, x3, x5, x6, x7, x8;
	if (x > 0 && x < (width - 1) && y>0 && y < (height - 1)) {
		x0 = img_in[(y - 1) * width + x - 1];
		x1 = img_in[(y - 1) * width + x];
		x2 = img_in[(y - 1) * width + x + 1];
		x3 = img_in[y * width + x - 1];

		x5 = img_in[y * width + x + 1];
		x6 = img_in[(y + 1) * width + x - 1];
		x7 = img_in[(y + 1) * width + x];
		x8 = img_in[(y + 1) * width + x + 1];

		G_x = (x0 + 2 * x3 + x6) - (x2 + 2 * x5 + x7);
		G_y = (x0 + 2 * x1 + x2) - (x6 + 2 * x7 + x8);
		img_out[idx] = (abs(G_x) + abs(G_y)) / 2;
	}

}


void print_img(const int* img, const int height, const int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			printf("%d ", img[y * width + x]);
		}
		printf("\n");
	}
}



int main(void) {

	const int KERNEL_SIZE = 3;
	int memsize_img = sizeof(int) * IMG_WIDTH * IMG_HEIGHT;
	int memsize_kernel = sizeof(int) * KERNEL_SIZE * KERNEL_SIZE;

	//cpu
	int* h_img_in, * h_img_out, *h_kernel;
	cudaMallocHost((void**)&h_img_in, memsize_img);
	cudaMallocHost((void**)&h_img_out, memsize_img);
	cudaMallocHost((void**)&h_kernel, memsize_kernel);

	//initialize
	for (int y = 0; y < IMG_HEIGHT; y++) {
		for (int x = 0; x < IMG_WIDTH; x++) {
			h_img_in[y * IMG_WIDTH + x] = rand() % 3+1;
		}
	}

	for (int y = 0; y < KERNEL_SIZE; y++) {
		for (int x = 0; x < KERNEL_SIZE; x++) {
			h_kernel[y * KERNEL_SIZE + x] = rand() % 2+1;
		}
	}

	//gpu
	int* d_img_in, * d_img_out, *d_kernel;
	cudaMalloc((void**)&d_img_in, memsize_img);
	cudaMalloc((void**)&d_img_out, memsize_img);
	cudaMalloc((void**)&d_kernel, memsize_kernel);
	cudaMemcpy(d_img_in, h_img_in, memsize_img, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kernel, h_kernel, memsize_kernel, cudaMemcpyHostToDevice);

	//grid and block
	unsigned int grid_size_row = CEIL(IMG_HEIGHT, BLOCK_SIZE);
	unsigned int grid_size_col = CEIL(IMG_WIDTH, BLOCK_SIZE);
	dim3 grid_size(grid_size_col, grid_size_row, 1);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
	//conv_k3 <<<grid_size, block_size>>> (d_img_in, d_kernel, d_img_out, IMG_HEIGHT, IMG_WIDTH);
	sobel_ex <<<grid_size, block_size>>> (d_img_in, d_img_out, IMG_HEIGHT, IMG_WIDTH);

	//test
	cudaMemcpy(h_img_out, d_img_out, memsize_img, cudaMemcpyDeviceToHost);
	printf("The input img is as follow: \n");
	print_img(h_img_in, IMG_HEIGHT, IMG_WIDTH);	
	printf("The kernel is as follow: \n");
	print_img(h_kernel, KERNEL_SIZE, KERNEL_SIZE);
	printf("The output img is as follow: \n");
	print_img(h_img_out, IMG_HEIGHT, IMG_WIDTH);


	free(h_img_in);
	free(h_img_out);
	free(h_kernel);
	cudaFree(d_img_in);
	cudaFree(d_img_out);
	cudaFree(d_kernel);

	return 0;
}
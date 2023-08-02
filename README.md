# CUDA-Learning
Using CUDA programming to implement some simple examples.
> The author conducts CUDA programming study based on some examples, aiming to understand some syntax operations as well as familiarize with introductory examples.
>
> Reference content:
>
> - [CUDA 编程模型系列五](https://www.bilibili.com/video/BV1vP411v7g4/?spm_id_from=333.788&vd_source=f9a58fa9ec474778cd43832fb746c14a)
> - [CUDA-Programming-book draft](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/book_draft.pdf)
>
> Table of Contents:
>
>  - [CUDA-Learning](#cuda-learning)
>  - [Mastery](#mastery)
>  - [Instance logging](#instance-logging)
>    - [`VecAdd.cu`](#vecaddcu)
>      - [memory type](#memory-type)
>      - [CHECK](#check)
>    - [`MM.cu`](#mmcu)
>      - [event time test](#event-time-test)
>    - [`conv.cu`](#convcu)
>  - [TO DO](#to-do)
>
> 2023-7-31--2023-8-2



## Mastery

- Building simple code frameworks

  > A coding framework is considered in terms of the following steps
  >
  > ```c++
  > int main(void){
  > //cpu & initialize
  > //gpu & initialize
  > //cpu to gpu
  > //calculate
  > //gpu to cpu
  > //test
  > //free
  > return 0; 
  > }
  > ```
  
- Some code details and tips

  > - Different types of memory in CUDA
  > - CHECK
  > - Event Timing

## Instance logging

### `VecAdd.cu`

This example implements the addition of two vectors, applying a one-dimensional `grid` and `block`. Here, the tutorial uses a `for` loop to test if the kernel function is correct. Given the parallel environment, the author considered why not just do the verification in parallel.

So it was adapted, but there were many errors, which are documented here, and the main function part is omitted below.

```c++

bool test_cpu = false;		//global variable
__global__ void check_gpu(const double* d_z, const double* gpu_result, const int N) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N) {
		if (fabs(d_z[index] - gpu_result[index]) > 1.0e-15) {
			test_cpu = true;
		}
	}
}
```

> The error is as follows:
>
> ```powershell
> error: identifier "test_cpu" is undefined in device code
> ```

Follow the prompt to change to:

```c++
__device__ bool test_cpu = false;
```

> The error is as follows:
>
> ```powershell
> warning #20091-D: a __device__ variable "test_cpu" cannot be directly read in a host function
> ```

So the author thought of transferring the variable ``test_cpu`` to the host and then outputting it, that is, forcing the conversion with the existing knowledge:

```c++
bool* error_cpu = (bool*)malloc(sizeof(bool));
cudaMemcpy(error_cpu, *test_cpu, sizeof(bool), cudaMemcpyDeviceToHost);
```

> The error is reported as follows:
>
> ```powershell
> error: operand of "*" must be a pointer but has type "__nv_bool"
> ```

Eventually looked up the new function and how to use it, and learned the memory types, which are listed below.

#### memory type

> Post the content here, haven't applied it, not sure of the exact call scenario and the way, which is also mentioned in **TO DO**.

![image](imgs/GPUmem2.png)

![image](imgs/GPUmem1.png)

The specific functions are found in the `MemTest.cu` file

> Use `printf( ) ` as an example. When it is placed under the `__global__` modifier, it acts as a function of the device, and when it is placed in the main function it is a function of the host.

#### CHECK

> Defines the `error.cuh` header file for error message output, which is used to check for the memory allocation and transfer problems described above.


### `MM.cu`

> This example implements multiplication of matrices, which in the tutorial are set to be square matrices, and is implemented here using `M`, `K`, and `N` with three different `row` and `col`. But only the naive version was implemented, without optimization for access etc. This is the next step in the program.

When I reproduced this example, I made some mistakes by not aligning the `x y` axes with the `col` and `row` of the matrix, but after fixing it, I also successfully implemented matrix multiplication and gained a better understanding of the assignment of `grid` and `block`.

#### event time test

> Test the runtime of the CPU version and the GPU version.

Some function templates are

```c++
int main(void){
  //define
  cudaEvent_t start, stop_cpu, stop_gpu;
  cudaEventCreate(&start);
  cudaEventCreate(&stop_cpu);
  cudaEventCreate(&stop_gpu);    

  cudaEventRecord(start);
  //GPU Function;
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  //CPU Function;
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
}	
```

                                           

### `conv.cu`

> This example implements the `3*3` convolution operation and `Sobel` edge detection, using `opencv`. The author tried unsuccessfully for 2 hours on a win11 system and successfully gave up, using random numbers to construct the grayscale image matrix.

The details and the configuration of `grid` and `block` are similar and not listed here.

## TO DO

- Optimizing Matrix Multiplication
- Matrix Transpose
- Atomic manipulation/reduction
- ......






  

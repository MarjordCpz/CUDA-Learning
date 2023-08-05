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


### `MM_V2.cu`

> Increase speed by chunking matrices and moving them to shared mem for multiplication.

In this implementation of matrix multiplication, a `BLOCK_SIZE` sized `sub` matrix is used to move the data. In this process, I wrote the wrong judgment condition of the loop and caused an error, which is recorded here.

When moving data, the variable `step` is used to split the original large matrix, and the loop condition of `step` is `step <= K / BLOCK_SIZE`. The condition `=` is not added because only the case of integer division is considered at first. And no out-of-bounds judgment was added, resulting in non-integer divisions not being written in the `sub` square.

**A few more details**.

- On Win system, add `cudaDeviceSynchronize();` after calling or executing the function of device to avoid the program can't output. Since this version uses ` __managed__`-modified variables, it does not use data-moving functions. The data moving function automatically calls `cudaDeviceSynchronize();`, so the previous version automatically adjusted it even if it didn't add this function.
- When timing events, two device function calls are timed using two sets of events.

The speed of the run has increased, and the result on one occasion was as follows

```powershell
Time_gpu1 is 1.73488 ms.
Time_gpu2 is 0.31091 ms.
Time_cpu is 171.48985 ms.
Result is pass.
```



### ``TransM.cu``

> Implement the matrix transpose operation, also use `shared mem` to speed up, but need to pay attention to avoid bank conflict problem. The author lists his understanding.

In shared memory, consecutive 32-bits words are allocated to 32 consecutive banks, which is like the seats in a movie theater: a column of seats is equivalent to a bank, so each row has 32 seats, and in each seat, you can "sit" a 32-bit data (or multiple data less than 32-bits, e.g., 4 data). bits of data, such as 4 char-type data, 2 short-type data); and under normal circumstances, we are in accordance with the order of the first line and then sit down a line to sit in the order of the seat, in the shared memory address mapping is the same way.

Multiple threads in a warp accessing the same bank will conflict, ignoring of course the multicast and broadcast mechanisms. Then as in the following figure, `bank[0][1]` and `bank[0][1]` are in conflict, and after the conflict, the parallelism will be converted to serialization, and the computational efficiency will be reduced dramatically.

Then using padding, you can add out of that column all shift, stagger the conflict data. As shown in the figure below.

! [](imgs/bank.svg)

The result is as follows. Shared memory is used to achieve the speedup effect.

```powershell
Time_gpu1 is 1.60374 ms.
Time_gpu2 is 0.71379 ms.
Time_cpu is 21.53981 ms.
Result is pass.
```


## TO DO

- Atomic manipulation/reduction
- ......






  

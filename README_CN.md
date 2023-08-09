# CUDA学习

> 笔者根据一些实例进行CUDA编程学习，旨在了解一些语法操作以及熟悉入门实例。
>
> 参考内容：
>
> - [CUDA编程模型系列](https://www.bilibili.com/video/BV1vP411v7g4/?spm_id_from=333.788&vd_source=f9a58fa9ec474778cd43832fb746c14a)
> - [CUDA-Programming-book draft](https://github.com/brucefan1983/CUDA-Programming/blob/master/src/book_draft.pdf)
>
> 目录
>
>  [CUDA学习](#cuda学习)
>  - [掌握的内容](#掌握的内容)
>  - [实例记录](#实例记录)
>    - [`VecAdd.cu`](#vecaddcu)
>      - [内存类型](#内存类型)
>      - [CHECK](#check)
>    - [`MM.cu`](#mmcu)
>      - [事件时间测试](#事件时间测试)
>    - [`conv.cu`](#convcu)
>    - [`MM_V2.cu`](#mm_v2cu)
>    - [`TransM.cu`](#transmcu)
>    - [`Reduce.cu`](#reducecu)
>    - [`topk.cu`](#topkcu)
> 
>
> 2023-7-31——2023-8-9



## 掌握的内容

- 构建简单的代码框架

  > 代码框架从以下几个步骤考虑
  >
  > ```c++
  > int main(void){
  >     //cpu & initialize
  >     //gpu & initialize
  >     //cpu to gpu
  >     //calculate
  >     //gpu to cpu
  >     //test
  >     //free
  >     return 0;
  > }
  > ```

- 一些代码细节和技巧

  > - CUDA中不同类型的内存
  > - CHECK
  > - 事件计时

## 实例记录

### `VecAdd.cu`

本实例实现两个向量相加，应用了一维的 `grid` 和 `block`。在这里，教程中使用的是 `for` 循环测试核函数是否正确。鉴于在并行的环境下，笔者考虑为什么不直接并行地进行验证呢？

于是就进行了改造，但是出现了许多错误，在此记录，以下省略主函数部分。

```c++
/*Ver1*/
bool test_cpu = false;		//全局变量
__global__ void check_gpu(const double* d_z, const double* gpu_result, const int N) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N) {
		if (fabs(d_z[index] - gpu_result[index]) > 1.0e-15) {
			test_cpu = true;
		}
	}
}
```

> 报错如下：
>
> ```powershell
> error: identifier "test_cpu" is undefined in device code
> ```

根据提示更改为：

```c++
__device__ bool test_cpu = false;
```

> 报错如下：
>
> ```powershell
> warning #20091-D: a __device__ variable "test_cpu" cannot be directly read in a host function
> ```

于是笔者想到将变量 `test_cpu` 转移到host上再输出，就是用现有的知识强行转换：

```c++
bool* error_cpu = (bool*)malloc(sizeof(bool));
cudaMemcpy(error_cpu, *test_cpu, sizeof(bool), cudaMemcpyDeviceToHost);
```

> 报错如下：
>
> ```powershell
> error: operand of "*" must be a pointer but has type "__nv_bool"
> ```

最终查到了新的函数以及使用方式，并且学习到了内存类型，如下列出。

#### 内存类型

> 将内容贴在这里，没有应用过，不清楚具体的调用情景以及方式，这也是 **TO DO** 里提到的。

![image](imgs/GPUmem2.png)

![image](imgs/GPUmem1.png)

具体的函数见 `MemTest.cu` 文件

> 以 `printf( ) `作为例子。当其放在`__global__`修饰的函数下，就作为device的函数，而放入main函数中就是host的函数。

#### CHECK

> 定义了 `error.cuh` 的头文件进行错误信息的输出，这个头文件用于检查上述的内存分配和转移问题。



### `MM.cu`

> 本实例实现矩阵的乘法，教程中的矩阵设定为方阵，这里使用 `M`，`K`，`N`三种不同的 `row` 和 `col` 进行实现。但是只实现了naive 的版本，没有对访存等进行优化。这是下一步计划。

笔者在进行这个实例的复现时，由于没有将`x y`轴与矩阵的 `col` 与 `row` 统一起来，导致出现了一些错误，进行修正后也顺利实现了矩阵乘法以及对 `grid` 和 `block` 的分配有了进一步的理解。

#### 事件时间测试

> 分别测试CPU版本和GPU版本的运行时间

一些函数模板如下

```c++
int main(void){
    //定义
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

​                                           

### `conv.cu`

> 该实例实现了 `3*3` 卷积操作以及 `Sobel` 边缘检测，要用到`opencv`。笔者在win11系统上尝试了2小时未成功，放弃，使用随机数模拟构造灰度图像矩阵。

具体细节以及grid和block的配置类似，不在这里列出。

### `MM_V2.cu`

> 将矩阵分块后搬移到shared mem中再进行相乘提高运行速度。

在这个矩阵相乘的实现中，使用了 `BLOCK_SIZE` 大小的 `sub` 方阵进行数据的搬移。在此过程中，笔者因为写错循环的判断条件而导致出错，在此记录。

在进行数据搬移时，会使用变量 `step` 将原始的大矩阵进行分割，`step` 的循环条件是 `step <= K / BLOCK_SIZE ` 。由于刚开始只考虑到了整除的情况，所以没有加条件 `=` 。并且没有加入越界判断，导致非整除的部分不能写道`sub`方阵中。

**还有的一些细节**

- 在win系统上，要在调用或执行完device的函数后加上 `cudaDeviceSynchronize();` 以免程序无法输出。由于这个版本使用的是 ` __managed__`修饰的变量，没有使用数据搬移的函数。而数据搬移的函数自动调用了 `cudaDeviceSynchronize();` ，因此上一个版本即使没有加这个函数，也会自动进行调整。
- 在对事件进行计时时，两个device函数的调用要使用两组事件进行计时。

运行速度提高了，某次结果如下

```powershell
Time_gpu1 is 1.73488 ms.
Time_gpu2 is 0.31091 ms.
Time_cpu is 171.48985 ms.
Result is pass.
```



### `TransM.cu`

> 实现矩阵的转置操作，同样还是利用 `shared mem` 进行提速，但是需要注意的是要避免bank冲突问题。笔者将自己的理解列出。

在共享内存中，连续的32-bits字被分配到连续的32个bank中，这就像电影院的座位一样：一列的座位就相当于一个bank，所以每行有32个座位，在每个座位上可以“坐”一个32-bits的数据(或者多个小于32-bits的数据，如4个char型的数据，2个short型的数据)；而正常情况下，我们是按照先坐完一行再坐下一行的顺序来坐座位的，在shared memory中地址映射的方式也是这样的。

一个warp中多个thread访问同一个bank会发生冲突，当然，在这里忽略multicast以及broadcast机制。那么如下图而言`bank[0][1]`和`bank[0][1]`时发生冲突的，冲突后，并行将转为串行，运算效率大幅下降。

那么采用padding的方式，可以将补充出来的那一列全部移位，错开发生冲突的数据。如下图所示。

![](imgs/bank.svg)

运行结果如下，使用了共享内存达到了加速效果。

```powershell
Time_gpu1 is 1.60374 ms.
Time_gpu2 is 0.71379 ms.
Time_cpu is 21.53981 ms.
Result is pass.
```


### `Reduce.cu`

> 实现了一个数组的求和操作，使用了共享内存以及原子操作，了解到该场景下的原子操作使用条件。记录核心步骤的理解。

- 将线程数能处理之外的数据加回到对应的位置，数据的间隔为所有的线程数。现在的数据长度是全部的线程数。
- 使用共享内存收集一个块内线程数大小的数据进行处理。现在相当于所有的块并行处理一个长度为 `blockDim.x` 的数据。
- 最后由于结果只有一个，每一个块的第一个线程使用原子相加的操作实现结果的累加。

笔者当初不理解为什么只申请了 `BLOCK_SIZE` 大小的共享内存，却可以将长度为 `gridDim.x * blcokDim.x` 的数据存放起来。最后想到了共享内存在每一个block中都有分布，因此，相当于是一个数组进行了并行运算。

### `topk.cu`

> 实现了找出前k个最大数据的功能。记录核心步骤的理解，如下。

- 在主函数的部分使用了 `2_pass` 将结果分成两步骤计算出来。

- 还是如同上一个规约的例子，先计算超出线程数的部分，按照大小排列好。现在的总数量为`gridDim.x * blcokDim.x` 。

- 分配共享内存，共享内存的大小为 `K * BLOCK_SIZE` 相当于每一个线程处理K个数据。为了便于理解这个过程，笔者是用少量数据测试了一下。

  ```powershell
  topk[0] is 15.
  topk[0] is 16.
  topk[0] is 17.
  topk[0] is 18.
  topk[0] is 19.
  topk[0] is 20.
  The blockidx is 1. Threadidx is 0.
  The blockidx is 1. Threadidx is 1.
  The blockidx is 2. Threadidx is 0.
  The blockidx is 2. Threadidx is 1.
  The blockidx is 0. Threadidx is 0.
  The blockidx is 0. Threadidx is 1.
  topk[1] is 9.
  topk[1] is 10.
  topk[1] is 11.
  topk[1] is 12.
  topk[1] is 13.
  topk[1] is 14.
  The blockidx is 1. Threadidx is 0.
  The blockidx is 1. Threadidx is 1.
  The blockidx is 2. Threadidx is 0.
  The blockidx is 2. Threadidx is 1.
  The blockidx is 0. Threadidx is 0.
  The blockidx is 0. Threadidx is 1.
  topk[2] is 3.
  topk[2] is 4.
  topk[2] is 5.
  topk[2] is 6.
  topk[2] is 7.
  topk[2] is 8.
  The blockidx is 1. Threadidx is 0.
  The blockidx is 1. Threadidx is 1.
  The blockidx is 2. Threadidx is 0.
  The blockidx is 2. Threadidx is 1.
  The blockidx is 0. Threadidx is 0.
  The blockidx is 0. Threadidx is 1.
  ```

  可以看到每一个线程管理k个数据，相当于上述的归约操作一个线程管理了一个数组。

- 仍然使用一个 `step` 进行循环，将原本thread对应的数组偏移，从中选取数据与原thread的数据进行排序，得到最终的结果

- 处理好以后可以得出每一个block下面的topk数组，使用 `2_pass` ，第二次调用函数时将 `grid_size`设置为1。

最终得出结果：

```powershell
Result: Pass
CPU time: 56.72; GPU time: 0.83
```


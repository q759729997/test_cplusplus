# linux使用openmp

- 头文件 `#include <omp.h>`
- 寻找可并行代码块 “structured block”:
- 使用openMP标记一行或多行语句，成为一个结构化模块，使用编译器指导语句并行执行.

## 依赖环境

- gcc
- gcc参数详解：<https://www.runoob.com/w3cnote/gcc-parameter-detail.html>

## 编译执行

- 文件目录

~~~shell
cd /media/sf_vbshare/github/test_cplusplus/ucas/openmp
~~~

- 普遍编译

~~~shell
gcc foo.c
./a.out
gcc hello.c&&./a.out
~~~

- 多进程编译

~~~shell
export OMP_NUM_THREADS=4

gcc -fopenmp foo.c
./a.out
gcc -fopenmp hello_par.c&&./a.out
~~~

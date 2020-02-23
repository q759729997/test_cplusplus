# linux使用openmp

## 依赖环境

- gcc

## 编译执行

- 文件目录

~~~shell
cd /media/sf_vbshare/github/test_cplusplus/ucas/openmp
~~~

- 普遍编译

~~~shell
gcc foo.c
./a.out
~~~

- 多进程编译

~~~shell
export OMP_NUM_THREADS=4

gcc -fopenmp foo.c
./a.out
~~~

# linux使用MPICH

## 环境设置

- 安装相关包:

~~~shell
apt-get install mpich
~~~

## 编译

- 代码位置：

~~~wiki
/media/sf_vbshare/github/test_cplusplus/ucas/MPI
~~~

- C:

~~~shell
mpicc test.c -o test
mpicc test.c -o test -lm # 动态链接库,示例为链接数学库
~~~

- C++：

~~~shell
mpicxx test.cpp -o test
~~~

## 运行

- 在本地节点用几个进程运行

~~~shell
mpiexec -n 16 ./test
~~~

- 多节点运行

~~~shell
mpiexec -hosts h1,h2,h3,h4 -n 16 ./test  # 自动分配
mpiexec -hosts h1:4,h2:4,h3:4,h4:4 -n 16 ./test  # 手工分配
~~~

- 通过host file文件预定义好节点

~~~shell
cat hf

h1:4
h2:2

mpiexec -hostfile hf -n 16 ./test
~~~

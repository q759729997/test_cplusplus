#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define  MASTER		0

int main (int argc, char *argv[])
{	// 程序1	
	// 编译：	mpicc mpi_hello.c -o hello
	// 运行：	mpiexec -n 2 ./hello
	int  numtasks, taskid, len;
	char hostname[MPI_MAX_PROCESSOR_NAME];  // 计算机名字

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	MPI_Get_processor_name(hostname, &len);
	
	printf ("Hello from task %d on %s!\n", taskid, hostname);
	
	if (taskid == MASTER)  // 主进程打印出来
	   printf("MASTER: Number of MPI tasks is: %d\n",numtasks);
   
	MPI_Finalize();
}
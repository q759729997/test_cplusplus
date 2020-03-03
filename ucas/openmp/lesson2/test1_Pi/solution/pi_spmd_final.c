/*

NAME:   PI SPMD final version without false sharing

This program will numerically compute the integral of

                  4/(1+x*x) 
				  
from 0 to 1.  The value of this integral is pi -- which 
is great since it gives us an easy way to check the answer.

The program was parallelized using OpenMP and an SPMD 
algorithm.  The following OpenMP specific lines were 
added: 

(1) A line to include omp.h -- the include file that 
contains OpenMP's function prototypes and constants.

(2) A pragma that tells OpenMP to create a team of threads
with an integer variable i being created for each thread.

(3) two function calls: one to get the thread ID (ranging
from 0 to one less than the number of threads), and the other
returning the total number of threads.

(4) A "single" construct so only one thread prints the number
of threads.

(5) A cyclic distribution of the loop by changing loop control
expressions to run from the thread ID incremented by the number 
of threads.  Local sums accumlated into sum[id].

(6) A barrier to make sure everyone's done.

(7) A single construct so only one thread combines the local
sums into a single global sum.

Note that this program avoids the false sharing problem
by storing partial sums into a private scalar.

History: Written by Tim Mattson, 11/99.

*/

#include <stdio.h>
#include <omp.h>

#define MAX_THREADS 4

static long num_steps = 100000000;
double step;
int main ()
{
	int i,j;
	double pi, full_sum = 0.0;
	double start_time, run_time;
	double sum[MAX_THREADS];

	step = 1.0/(double) num_steps;

	for(j = 1; j <= MAX_THREADS; j++)
	{
		omp_set_num_threads(j);
		full_sum = 0.0;
		start_time = omp_get_wtime();

		#pragma omp parallel private(i)  // i为每个线程私有
		{
			int id = omp_get_thread_num();
			int numthreads = omp_get_num_threads();
			double x;

			double partial_sum = 0;

			#pragma omp single  // 默认有barrier（栅栏）
				printf(" num_threads = %d",numthreads);

			for (i = id; i < num_steps; i += numthreads){
				x = (i+0.5)*step;
				partial_sum += + 4.0/(1.0+x*x);
			}
				
			#pragma omp critical  // 互斥，一次只有一个线程可以进入该区域，π的更新不冲突
				full_sum += partial_sum;
		}
		
		pi = step * full_sum;
		run_time = omp_get_wtime() - start_time;
		printf("\n pi is %f in %f seconds %d threds \n ",pi,run_time,j);
	}
}
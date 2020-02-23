# include<stdio.h>
#include<omp.h>

int main()
{
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();
        printf("hello(%d)", ID);
        printf("world(%d)\n", ID);
        /*
        hello(1)world(1)
        hello(2)world(2)
        hello(3)world(3)
        hello(0)world(0)
        */
    }
    return 0;
}
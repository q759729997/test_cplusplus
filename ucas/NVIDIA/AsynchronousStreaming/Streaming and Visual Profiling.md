<h1><div align="center">Asynchronous Streaming, and Visual Profiling with CUDA C/C++</div></h1>

![CUDA](./images/CUDA_Logo.jpg)

The CUDA tookit ships with the **Nsight Systems**, a powerful GUI application to support the development of accelerated CUDA applications. Nsight Systems generates a graphical timeline of an accelerated application, with detailed information about CUDA API calls, kernel execution, memory activity, and the use of **CUDA streams**.

In this lab, you will be using the Nsight Systems timeline to guide you in optimizing accelerated applications. Additionally, you will learn some intermediate CUDA programming techniques to support your work: **unmanaged memory allocation and migration**; **pinning**, or **page-locking** host memory; and **non-default concurrent CUDA streams**.

At the end of this lab, you will be presented with an assessment, to accelerate and optimize a simple n-body particle simulator, which will allow you to demonstrate the skills you have developed during this course. Those of you who are able to accelerate the simulator while maintaining its correctness, will be granted a certification as proof of your competency.

---
## Prerequisites

To get the most out of this lab you should already be able to:

- Write, compile, and run C/C++ programs that both call CPU functions and launch GPU kernels.
- Control parallel thread hierarchy using execution configuration.
- Refactor serial loops to execute their iterations in parallel on a GPU.
- Allocate and free CUDA Unified Memory.
- Understand the behavior of Unified Memory with regard to page faulting and data migrations.
- Use asynchronous memory prefetching to reduce page faults and data migrations.

## Objectives

By the time you complete this lab you will be able to:

- Use **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
- Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
- Utilize CUDA streams for concurrent kernel execution in accelerated applications.
- (**Optional Advanced Content**) Use manual device memory allocation, including allocating pinned memory, in order to asynchronously transfer data in concurrent CUDA streams.

---
## Running Nsight Systems

For this interactive lab environment, we have set up a remote desktop you can access from your browser, where you will be able to launch and use Nsight Systems.

You will begin by creating a report file for an already-existing vector addition program, after which you will be walked through a series of steps to open this report file in Nsight Systems, and to make the visual experience nice.

### Generate Report File

[`01-vector-add.cu`](../edit/01-vector-add/01-vector-add.cu) (<-------- click on these links to source files to edit them in the browser) contains a working, accelerated, vector addition application. Use the code execution cell directly below (you can execute it, and any of the code execution cells in this lab by `CTRL` + clicking it) to compile and run it. You should see a message printed that indicates it was successful.


```python
!nvcc -o vector-add-no-prefetch 01-vector-add/01-vector-add.cu -run
```

    Success! All values calculated correctly.


Next, use `nsys profile --stats=true` to create a report file that you will be able to open in the Nsight Systems visual profiler. Here we use the `-o` flag to give the report file a memorable name:


```python
!nsys profile --stats=true -o vector-add-no-prefetch-report ./vector-add-no-prefetch
```

    
    **** collection configuration ****
    	output_filename = /dli/task/vector-add-no-prefetch-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./vector-add-no-prefetch
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    Success! All values calculated correctly.
    	Generating the /dli/task/vector-add-no-prefetch-report.qdstrm file.
    	Capturing raw events...
    	10720 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/vector-add-no-prefetch-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/vector-add-no-prefetch-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 10678 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/vector-add-no-prefetch-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       61.3       195398538           3      65132846.0           26292       195325939  cudaMallocManaged                                                               
       31.4        99963724           1      99963724.0        99963724        99963724  cudaDeviceSynchronize                                                           
        7.3        23406080           3       7802026.7         7076571         9120337  cudaFree                                                                        
        0.0           62949           1         62949.0           62949           62949  cudaLaunchKernel                                                                
    
    
    
    
    Generating CUDA Kernel Statistics...
    
    Generating CUDA Memory Operation Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0        99957801           1      99957801.0        99957801        99957801  addVectorsInto                                                                  
    
    
    CUDA Memory Operation Statistics (nanoseconds)
    
    Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       81.6        50373408        8629          5837.7            2752           91616  [CUDA Unified Memory memcpy HtoD]                                               
       18.4        11358720         768         14790.0            1824           82432  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    CUDA Memory Operation Statistics (KiB)
    
                Total      Operations            Average            Minimum            Maximum  Name                                                                            
    -----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
             308316.0            8629               35.7              4.000              960.0  [CUDA Unified Memory memcpy HtoD]                                               
             131072.0             768              170.7              4.000             1020.0  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       55.5      1518925059          81      18752161.2            2166       100188234  poll                                                                            
       40.8      1116220880          81      13780504.7           24358       100123797  sem_timedwait                                                                   
        2.7        73903640         577        128082.6            1029        15889342  ioctl                                                                           
        0.9        25749151          87        295967.3            1707         9066010  mmap                                                                            
        0.0          678507          96          7067.8            1693           29241  fopen                                                                           
        0.0          501811          78          6433.5            4361           13844  open64                                                                          
        0.0          255473          89          2870.5            1591            4858  fclose                                                                          
        0.0          143437           4         35859.3           32679           41462  pthread_create                                                                  
        0.0          113081          10         11308.1            7325           14273  write                                                                           
        0.0           92653           3         30884.3           24197           42400  fgets                                                                           
        0.0           90653          81          1119.2            1002            4256  fcntl                                                                           
        0.0           43184          13          3321.8            2106            5348  munmap                                                                          
        0.0           37027           5          7405.4            4447           11558  open                                                                            
        0.0           27433          12          2286.1            1368            3948  read                                                                            
        0.0           17301           3          5767.0            5031            6374  pipe2                                                                           
        0.0           12580           2          6290.0            5759            6821  socket                                                                          
        0.0            8983           4          2245.8            1919            2560  mprotect                                                                        
        0.0            7691           1          7691.0            7691            7691  connect                                                                         
        0.0            7301           2          3650.5            3131            4170  fread                                                                           
        0.0            6854           1          6854.0            6854            6854  pthread_cond_broadcast                                                          
        0.0            2716           1          2716.0            2716            2716  bind                                                                            
        0.0            2149           1          2149.0            2149            2149  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


### Open the Remote Desktop

Run the next cell to generate a link to a remote desktop then read the instructions that follow.


```python
%%js
var url = window.location.hostname + ':5901';
element.append('<a href="'+url+'">Open Remote Desktop</a>')
```


    <IPython.core.display.Javascript object>


After clicking the _Connect_ button you will be asked for a password, which is `nvidia`.

### Open the Remote Desktop Terminal Application

Next, click on the icon for the terminal application, found at the bottom of the screen of the remote desktop:

![terminal](images/terminal.png)

### Open Nsight Systems

To open Nsight Systems, enter and run the `nsight-sys` command from the now-open terminal:

![open nsight](images/open-nsight.png)

### Enable Usage Reporting

When prompted, click "Yes" to enable usage reporting:

![enable usage](images/enable_usage.png)

### Open the Report File

Open this report file by visiting _File_ -> _Open_ from the Nsight Systems menu, then go to the path `/root/Desktop/reports` and select `vector-add-no-prefetch-report.qdrep`. All the reports you generate in this lab will be in this `root/Desktop/reports` directory:

![open-report](images/open-report.png)

### Ignore Warnings/Errors

You can close and ignore any warnings or errors you see, which are just a result of our particular remote desktop environment:

![ignore errors](images/ignore-error.png)

### Make More Room for the Timelines

To make your experience nicer, full-screen the profiler, close the _Project Explorer_ and hide the *Events View*:

![make nice](images/make-nice.png)

Your screen should now look like this:

![now nice](images/now-nice.png)

### Expand the CUDA Unified Memory Timelines

Next, expand the _CUDA_ -> _Unified memory_ and _Context_ timelines, and close the _OS runtime libraries_ timelines:

![open memory](images/open-memory.png)

### Observe Many Memory Transfers

From a glance you can see that your application is taking about 1 second to run, and that also, during the time when the `addVectorsInto` kernel is running, that there is a lot of UM memory activity:

![memory and kernel](images/memory-and-kernel.png)

Zoom into the memory timelines to see more clearly all the small memory transfers being caused by the on-demand memory page faults. A couple tips:

1. You can zoom in and out at any point of the timeline by holding `Ctrl` while scrolling your mouse/trackpad
2. You can zoom into any section by click + dragging a rectangle around it, and then selecting _Zoom in_

Here's an example of zooming in to see the many small memory transfers:

![many transfers](images/many-transfers.png)

---
## Comparing Code Refactors Iteratively with Nsight Systems

Now that you have Nsight Systems up and running and are comfortable moving around the timelines, you will be profiling a series of programs that were iteratively improved using techniques already familiar to you. Each time you profile, information in the timeline will give information supporting how you should next modify your code. Doing this will further increase your understanding of how various CUDA programming techniques affect application performance.

### Exercise: Compare the Timelines of Prefetching vs. Non-Prefetching

[`01-vector-add-prefetch-solution.cu`](../edit/01-vector-add/solutions/01-vector-add-prefetch-solution.cu) refactors the vector addition application from above so that the 3 vectors needed by its `addVectorsInto` kernel are asynchronously prefetched to the active GPU device prior to launching the kernel (using [`cudaMemPrefetchAsync`](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8dc9199943d421bc8bc7f473df12e42)). Open the source code and identify where in the application these changes were made.

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.


```python
!nvcc -o vector-add-prefetch 01-vector-add/solutions/01-vector-add-prefetch-solution.cu -run
```

    Success! All values calculated correctly.


Now create a report file for this version of the application:


```python
!nsys profile --stats=true -o vector-add-prefetch-report ./vector-add-prefetch
```

    
    **** collection configuration ****
    	output_filename = /dli/task/vector-add-prefetch-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./vector-add-prefetch
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    Success! All values calculated correctly.
    	Generating the /dli/task/vector-add-prefetch-report.qdstrm file.
    	Capturing raw events...
    	2293 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/vector-add-prefetch-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/vector-add-prefetch-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 2251 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/vector-add-prefetch-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       75.5       195006782           3      65002260.7           24596       194937140  cudaMallocManaged                                                               
       11.8        30454009           1      30454009.0        30454009        30454009  cudaDeviceSynchronize                                                           
        8.7        22522828           3       7507609.3         6743393         8911891  cudaFree                                                                        
        4.0        10310164           3       3436721.3            9571        10132579  cudaMemPrefetchAsync                                                            
        0.0           46578           1         46578.0           46578           46578  cudaLaunchKernel                                                                
    
    
    
    
    Generating CUDA Kernel Statistics...
    
    Generating CUDA Memory Operation Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0          505541           1        505541.0          505541          505541  addVectorsInto                                                                  
    
    
    CUDA Memory Operation Statistics (nanoseconds)
    
    Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       76.6        37238400         192        193950.0          192576          196448  [CUDA Unified Memory memcpy HtoD]                                               
       23.4        11393152         768         14834.8            1920           88448  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    CUDA Memory Operation Statistics (KiB)
    
                Total      Operations            Average            Minimum            Maximum  Name                                                                            
    -----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
             393216.0             192             2048.0           2048.000             2048.0  [CUDA Unified Memory memcpy HtoD]                                               
             131072.0             768              170.7              4.000             1020.0  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       53.7      1378938161          81      17023927.9            2971       100176989  poll                                                                            
       40.9      1048680083          74      14171352.5           23643       100125639  sem_timedwait                                                                   
        4.2       107566141         580        185458.9            1030        15454768  ioctl                                                                           
        1.0        24912208          88        283093.3            1698         8859946  mmap                                                                            
        0.1         3364954           3       1121651.3           64689         3094258  sem_wait                                                                        
        0.0          639482          96          6661.3            1616           14955  fopen                                                                           
        0.0          519274          78          6657.4            3808           13528  open64                                                                          
        0.0          259593          89          2916.8            1527            4257  fclose                                                                          
        0.0          203706           5         40741.2           34271           60974  pthread_create                                                                  
        0.0          145284          14         10377.4            1835           13962  write                                                                           
        0.0           91728           3         30576.0           23381           42151  fgets                                                                           
        0.0           90524          80          1131.5            1017            4834  fcntl                                                                           
        0.0           38256          11          3477.8            2353            4704  munmap                                                                          
        0.0           35843          16          2240.2            1394            4944  read                                                                            
        0.0           31785           5          6357.0            4366            9678  open                                                                            
        0.0           16995           3          5665.0            4978            6664  pipe2                                                                           
        0.0           13477           5          2695.4            1832            4247  mprotect                                                                        
        0.0           11081           2          5540.5            4719            6362  socket                                                                          
        0.0            8065           1          8065.0            8065            8065  connect                                                                         
        0.0            6854           2          3427.0            2703            4151  fread                                                                           
        0.0            6158           1          6158.0            6158            6158  pthread_cond_broadcast                                                          
        0.0            2784           1          2784.0            2784            2784  bind                                                                            
        0.0            2011           1          2011.0            2011            2011  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


Open the report in Nsight Systems, leaving the previous report open for comparison.

- How does the execution time compare to that of the `addVectorsInto` kernel prior to adding asynchronous prefetching?
- Locate `cudaMemPrefetchAsync` in the *CUDA API* section of the timeline.
- How have the memory transfers changed?


### Exercise: Profile Refactor with Launch Init in Kernel

In the previous iteration of the vector addition application, the vector data is being initialized on the CPU, and therefore needs to be migrated to the GPU before the `addVectorsInto` kernel can operate on it.

The next iteration of the application, [01-init-kernel-solution.cu](../edit/02-init-kernel/solutions/01-init-kernel-solution.cu), the application has been refactored to initialize the data in parallel on the GPU.

Since the initialization now takes place on the GPU, prefetching has been done prior to initialization, rather than prior to the vector addition work. Review the source code to identify where these changes have been made.

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.


```python
!nvcc -o init-kernel 02-init-kernel/solutions/01-init-kernel-solution.cu -run
```

    Success! All values calculated correctly.


Now create a report file for this version of the application:


```python
!nsys profile --stats=true -o init-kernel-report ./init-kernel
```

    
    **** collection configuration ****
    	output_filename = /dli/task/init-kernel-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./init-kernel
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    Success! All values calculated correctly.
    	Generating the /dli/task/init-kernel-report.qdstrm file.
    	Capturing raw events...
    	2012 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/init-kernel-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/init-kernel-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 1975 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/init-kernel-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       89.3       195007742           3      65002580.7           25407       194938413  cudaMallocManaged                                                               
        8.7        18998238           3       6332746.0         5170308         8561893  cudaFree                                                                        
        1.5         3214690           1       3214690.0         3214690         3214690  cudaDeviceSynchronize                                                           
        0.4          968037           3        322679.0           10710          848857  cudaMemPrefetchAsync                                                            
        0.0           63604           4         15901.0            9108           33619  cudaLaunchKernel                                                                
    
    
    
    
    Generating CUDA Kernel Statistics...
    
    Generating CUDA Memory Operation Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       51.5          501169           1        501169.0          501169          501169  addVectorsInto                                                                  
       48.5          472274           3        157424.7          154716          162523  initWith                                                                        
    
    
    CUDA Memory Operation Statistics (nanoseconds)
    
    Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0        11399904         768         14843.6            1920           88896  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    CUDA Memory Operation Statistics (KiB)
    
                Total      Operations            Average            Minimum            Maximum  Name                                                                            
    -----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
             131072.0             768              170.7              4.000             1020.0  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       45.3       474907155          30      15830238.5           23677       100121202  sem_timedwait                                                                   
       45.1       472851084          35      13510031.0            2834       100174334  poll                                                                            
        7.3        76008819         579        131276.0            1001        15693277  ioctl                                                                           
        2.0        21376650          88        242916.5            1730         8512663  mmap                                                                            
        0.1          647994          96          6749.9            1661           15851  fopen                                                                           
        0.1          641779           5        128355.8           34433          489132  pthread_create                                                                  
        0.0          479061          78          6141.8            3785           10928  open64                                                                          
        0.0          328465           3        109488.3           36737          196413  sem_wait                                                                        
        0.0          283642          89          3187.0            1630           24635  fclose                                                                          
        0.0          138478          14          9891.3            1734           14988  write                                                                           
        0.0           92530           3         30843.3           23909           41578  fgets                                                                           
        0.0           88320          79          1118.0            1004            4371  fcntl                                                                           
        0.0           39808          12          3317.3            2002            4981  munmap                                                                          
        0.0           35518          16          2219.9            1178            3856  read                                                                            
        0.0           31570           5          6314.0            4518            9036  open                                                                            
        0.0           15007           3          5002.3            4054            5891  pipe2                                                                           
        0.0           11813           5          2362.6            2117            2986  mprotect                                                                        
        0.0           11366           2          5683.0            4580            6786  socket                                                                          
        0.0            9326           3          3108.7            1939            4387  fread                                                                           
        0.0            8066           1          8066.0            8066            8066  connect                                                                         
        0.0            5799           1          5799.0            5799            5799  pthread_cond_broadcast                                                          
        0.0            2900           1          2900.0            2900            2900  bind                                                                            
        0.0            2173           1          2173.0            2173            2173  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


Open the new report file in Nsight Systems and do the following:

- Compare the application and `addVectorsInto` runtimes to the previous version of the application, how did they change?
- Look at the *Kernels* section of the timeline. Which of the two kernels (`addVectorsInto` and the initialization kernel) is taking up the majority of the time on the GPU?
- Which of the following does your application contain?
  - Data Migration (HtoD)
  - Data Migration (DtoH)

### Exercise: Profile Refactor with Asynchronous Prefetch Back to the Host

Currently, the vector addition application verifies the work of the vector addition kernel on the host. The next refactor of the application, [01-prefetch-check-solution.cu](../edit/04-prefetch-check/solutions/01-prefetch-check-solution.cu), asynchronously prefetches the data back to the host for verification.

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.


```python
!nvcc -o prefetch-to-host 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

    Success! All values calculated correctly.


Now create a report file for this version of the application:


```python
!nsys profile --stats=true -o prefetch-to-host-report ./prefetch-to-host
```

    
    **** collection configuration ****
    	output_filename = /dli/task/prefetch-to-host-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./prefetch-to-host
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    Success! All values calculated correctly.
    	Generating the /dli/task/prefetch-to-host-report.qdstrm file.
    	Capturing raw events...
    	1308 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/prefetch-to-host-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/prefetch-to-host-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 1271 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/prefetch-to-host-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       82.3       195632409           3      65210803.0           25687       195548607  cudaMallocManaged                                                               
        8.5        20121760           4       5030440.0           11888        19146464  cudaMemPrefetchAsync                                                            
        7.8        18587886           3       6195962.0         5170827         8073276  cudaFree                                                                        
        1.3         3207872           1       3207872.0         3207872         3207872  cudaDeviceSynchronize                                                           
        0.0           87590           4         21897.5           13309           41668  cudaLaunchKernel                                                                
    
    
    
    
    Generating CUDA Kernel Statistics...
    
    Generating CUDA Memory Operation Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       51.3          499862           1        499862.0          499862          499862  addVectorsInto                                                                  
       48.7          474101           3        158033.7          155239          162055  initWith                                                                        
    
    
    CUDA Memory Operation Statistics (nanoseconds)
    
    Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0        10646784          64        166356.0          166176          166752  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    CUDA Memory Operation Statistics (KiB)
    
                Total      Operations            Average            Minimum            Maximum  Name                                                                            
    -----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
             131072.0              64             2048.0           2048.000             2048.0  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       44.3       442128985          32      13816530.8            2564       100195301  poll                                                                            
       43.7       435717287          26      16758357.2           21674       100115988  sem_timedwait                                                                   
        9.6        95462497         583        163743.6            1033        19115892  ioctl                                                                           
        2.1        20939620          88        237950.2            1717         8022875  mmap                                                                            
        0.1          652018          96          6791.9            1752           20293  fopen                                                                           
        0.1          637770           5        127554.0           32430          482460  pthread_create                                                                  
        0.1          526186          78          6746.0            4162           13680  open64                                                                          
        0.0          383969           3        127989.7           47920          239524  sem_wait                                                                        
        0.0          259350          89          2914.0            1592            5301  fclose                                                                          
        0.0          140815          14         10058.2            1427           14749  write                                                                           
        0.0           92687           3         30895.7           24374           42199  fgets                                                                           
        0.0           90495          79          1145.5            1005            5414  fcntl                                                                           
        0.0           42702          13          3284.8            2062            5431  munmap                                                                          
        0.0           36420          16          2276.3            1248            4107  read                                                                            
        0.0           31408           5          6281.6            4582            9286  open                                                                            
        0.0           16643           3          5547.7            4024            6566  pipe2                                                                           
        0.0           12668           5          2533.6            2130            3352  mprotect                                                                        
        0.0           10312           2          5156.0            4444            5868  socket                                                                          
        0.0            8937           3          2979.0            1782            4192  fread                                                                           
        0.0            7485           1          7485.0            7485            7485  connect                                                                         
        0.0            6888           2          3444.0            1044            5844  pthread_cond_broadcast                                                          
        0.0            2822           1          2822.0            2822            2822  bind                                                                            
        0.0            2041           1          2041.0            2041            2041  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


Open this report file in Nsight Systems, and do the following:

- Use the *Unified Memory* section of the timeline to compare and contrast the *Data Migration (DtoH)* events before and after adding prefetching back to the CPU.

---
## Concurrent CUDA Streams

You are now going to learn about a new concept, **CUDA Streams**. After an introduction to them, you will return to using Nsight Systems to better evaluate their impact on your application's performance.

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.


```python
%%HTML

<div align="center"><iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRVgzpDzp5fWAu-Zpuyr09rmIqE4FTFESjajhfZSnY7yVvPgZUDxECAPSdLko5DZNTGEN7uA79Hfovd/embed?start=false&loop=false&delayms=3000" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>
```



<div align="center"><iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRVgzpDzp5fWAu-Zpuyr09rmIqE4FTFESjajhfZSnY7yVvPgZUDxECAPSdLko5DZNTGEN7uA79Hfovd/embed?start=false&loop=false&delayms=3000" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>



In CUDA programming, a **stream** is a series of commands that execute in order. In CUDA applications, kernel execution, as well as some memory transfers, occur within CUDA streams. Up until this point in time, you have not been interacting explicitly with CUDA streams, but in fact, your CUDA code has been executing its kernels inside of a stream called *the default stream*.

CUDA programmers can create and utilize non-default CUDA streams in addition to the default stream, and in doing so, perform multiple operations, such as executing multiple kernels, concurrently, in different streams. Using multiple streams can add an additional layer of parallelization to your accelerated applications, and offers many more opportunities for application optimization.

### Rules Governing the Behavior of CUDA Streams

There are a few rules, concerning the behavior of CUDA streams, that should be learned in order to utilize them effectively:

- Operations within a given stream occur in order.
- Operations in different non-default streams are not guaranteed to operate in any specific order relative to each other.
- The default stream is blocking and will both wait for all other streams to complete before running, and, will block other streams from running until it completes.

### Creating, Utilizing, and Destroying Non-Default CUDA Streams

The following code snippet demonstrates how to create, utilize, and destroy a non-default CUDA stream. You will note, that to launch a CUDA kernel in a non-default CUDA stream, the stream must be passed as the optional 4th argument of the execution configuration. Up until now you have only utilized the first 2 arguments of the execution configuration:

```cpp
cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.

someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.

cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
```

Outside the scope of this lab, but worth mentioning, is the optional 3rd argument of the execution configuration. This argument allows programmers to supply the number of bytes in **shared memory** (an advanced topic that will not be covered presently) to be dynamically allocated per block for this kernel launch. The default number of bytes allocated to shared memory per block is `0`, and for the remainder of the lab, you will be passing `0` as this value, in order to expose the 4th argument, which is of immediate interest:

### Exercise: Predict Default Stream Behavior

The [01-print-numbers](../edit/05-stream-intro/01-print-numbers.cu) application has a very simple `printNumber` kernel which accepts an integer and prints it. The kernel is only being executed with a single thread inside a single block, however, it is being executed 5 times, using a for-loop, and passing each launch the number of the for-loop's iteration.

Compile and run [01-print-numbers](../edit/05-stream-intro/01-print-numbers.cu) using the code execution block below. You should see the numbers `0` through `4` printed.


```python
!nvcc -o print-numbers 05-stream-intro/01-print-numbers.cu -run
```

    0
    1
    2
    3
    4


Knowing that by default kernels are executed in the default stream, would you expect that the 5 launches of the `print-numbers` program executed serially, or in parallel? You should be able to mention two features of the default stream to support your answer. Create a report file in the cell below and open it in Nsight Systems to confirm your answer.


```python
!nsys profile --stats=true -o print-numbers-report ./print-numbers
```

    
    **** collection configuration ****
    	output_filename = /dli/task/print-numbers-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./print-numbers
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    The application process terminated. One or more process it created re-parented. Waiting for termination of re-parented processes. To modify this behavior, use the `--wait` option.
     0
    1
    2
    3
    4
    	Generating the /dli/task/print-numbers-report.qdstrm file.
    	Capturing raw events...
    	1169 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/print-numbers-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/print-numbers-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 1133 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/print-numbers-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       99.9       179830163           5      35966032.6            6885       179796169  cudaLaunchKernel                                                                
        0.1          152668           1        152668.0          152668          152668  cudaDeviceSynchronize                                                           
    
    
    
    
    Generating CUDA Kernel Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0          174428           5         34885.6           32671           42303  printNumber                                                                     
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       44.7       211470525          13      16266963.5           21645       100115593  sem_timedwait                                                                   
       37.7       178559652          12      14879971.0           37050        78554334  poll                                                                            
       16.5        78159170         567        137846.9            1013        15509644  ioctl                                                                           
        0.6         2611289          80         32641.1            1688          780143  mmap                                                                            
        0.1          654194          97          6744.3            1832           16307  fopen                                                                           
        0.1          580867           4        145216.7           35287          470141  pthread_create                                                                  
        0.1          533473          79          6752.8            4106           13441  open64                                                                          
        0.1          258440          90          2871.6            1674            4332  fclose                                                                          
        0.0          125879          11         11443.5            7227           15600  write                                                                           
        0.0           96570          83          1163.5            1019            4535  fcntl                                                                           
        0.0           92460           3         30820.0           23170           42210  fgets                                                                           
        0.0           32747           5          6549.4            4438            9417  open                                                                            
        0.0           30246          13          2326.6            1365            4233  read                                                                            
        0.0           28300           1         28300.0           28300           28300  sem_wait                                                                        
        0.0           25046           7          3578.0            2240            5371  munmap                                                                          
        0.0           15614           3          5204.7            4153            6061  pipe2                                                                           
        0.0           11203           2          5601.5            4826            6377  socket                                                                          
        0.0            9059           4          2264.8            2022            2565  mprotect                                                                        
        0.0            7206           1          7206.0            7206            7206  connect                                                                         
        0.0            6948           2          3474.0            2797            4151  fread                                                                           
        0.0            5988           1          5988.0            5988            5988  pthread_cond_broadcast                                                          
        0.0            5594           2          2797.0            1070            4524  fflush                                                                          
        0.0            2720           1          2720.0            2720            2720  bind                                                                            
        0.0            2010           1          2010.0            2010            2010  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


### Exercise: Implement Concurrent CUDA Streams

Both because all 5 kernel launches occured in the same stream, you should not be surprised to have seen that the 5 kernels executed serially. Additionally you could make the case that because the default stream is blocking, each launch of the kernel would wait to complete before the next launch, and this is also true.

Refactor [01-print-numbers](../edit/05-stream-intro/01-print-numbers.cu) so that each kernel launch occurs in its own non-default stream. Be sure to destroy the streams you create after they are no longer needed. Compile and run the refactored code with the code execution cell directly below. You should still see the numbers `0` through `4` printed, though not necessarily in ascending order. Refer to [the solution](../edit/05-stream-intro/solutions/01-print-numbers-solution.cu) if you get stuck.


```python
!nvcc -o print-numbers-in-streams 05-stream-intro/01-print-numbers.cu -run
```

    0
    1
    2
    3
    4


Now that you are using 5 different non-default streams for each of the 5 kernel launches, do you expect that they will run serially or in parallel? In addition to what you now know about streams, take into account how trivial the `printNumber` kernel is, meaning, even if you predict parallel runs, will the speed at which one kernel will complete allow for complete overlap?

After hypothesizing, open a new report file in Nsight Systems to view its actual behavior. You should notice that now, there are additional rows in the _CUDA_ section for each of the non-default streams you created:


```python
!nsys profile --stats=true -o print-numbers-in-streams-report print-numbers-in-streams
```

    
    **** collection configuration ****
    	output_filename = /dli/task/print-numbers-in-streams-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = print-numbers-in-streams
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    The application process terminated. One or more process it created re-parented. Waiting for termination of re-parented processes. To modify this behavior, use the `--wait` option.
     0
    1
    2
    3
    4
    	Generating the /dli/task/print-numbers-in-streams-report.qdstrm file.
    	Capturing raw events...
    	1163 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/print-numbers-in-streams-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/print-numbers-in-streams-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 1127 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/print-numbers-in-streams-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       99.9       179912903           5      35982580.6            6920       179878647  cudaLaunchKernel                                                                
        0.1          151596           1        151596.0          151596          151596  cudaDeviceSynchronize                                                           
    
    
    
    
    Generating CUDA Kernel Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0          174269           5         34853.8           32703           42271  printNumber                                                                     
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       44.8       213347746          13      16411365.1           21399       100116453  sem_timedwait                                                                   
       37.6       179086124          12      14923843.7           34873        79103624  poll                                                                            
       16.6        78793152         564        139704.2            1010        15846751  ioctl                                                                           
        0.5         2590994          80         32387.4            1783          774945  mmap                                                                            
        0.1          671746          97          6925.2            1725           17912  fopen                                                                           
        0.1          501545          79          6348.7            4382           13558  open64                                                                          
        0.1          257079          90          2856.4            1560            4599  fclose                                                                          
        0.0          152466           4         38116.5           32462           42281  pthread_create                                                                  
        0.0          122479          11         11134.5            6952           14749  write                                                                           
        0.0           92965          82          1133.7            1011            4440  fcntl                                                                           
        0.0           90047           3         30015.7           21604           42528  fgets                                                                           
        0.0           32934           5          6586.8            4404            9356  open                                                                            
        0.0           30412          13          2339.4            1357            3546  read                                                                            
        0.0           25323           7          3617.6            2535            5150  munmap                                                                          
        0.0           16492           3          5497.3            4247            6430  pipe2                                                                           
        0.0           15694           2          7847.0            7789            7905  socket                                                                          
        0.0            9009           4          2252.2            2039            2445  mprotect                                                                        
        0.0            8861           1          8861.0            8861            8861  connect                                                                         
        0.0            6702           2          3351.0            2670            4032  fread                                                                           
        0.0            6609           1          6609.0            6609            6609  pthread_cond_broadcast                                                          
        0.0            3993           1          3993.0            3993            3993  fflush                                                                          
        0.0            2819           1          2819.0            2819            2819  bind                                                                            
        0.0            2184           1          2184.0            2184            2184  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


![streams overlap](images/streams-overlap.png)

### Exercise: Use Streams for Concurrent Data Initialization Kernels

The vector addition application you have been working with, [01-prefetch-check-solution.cu](../edit/04-prefetch-check/solutions/01-prefetch-check-solution.cu), currently launches an initialization kernel 3 times - once each for each of the 3 vectors needing initialization for the `vectorAdd` kernel. Refactor it to launch each of the 3 initialization kernel launches in their own non-default stream. You should still be see the success message print when compiling and running with the code execution cell below. Refer to [the solution](../edit/06-stream-init/solutions/01-stream-init-solution.cu) if you get stuck.


```python
!nvcc -o init-in-streams 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

    Success! All values calculated correctly.


Open a report in Nsight Systems to confirm that your 3 initialization kernel launches are running in their own non-default streams, with some degree of concurrent overlap.


```python
!nsys profile --stats=true -o init-in-streams-report ./init-in-streams
```

    
    **** collection configuration ****
    	output_filename = /dli/task/init-in-streams-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./init-in-streams
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    Success! All values calculated correctly.
    	Generating the /dli/task/init-in-streams-report.qdstrm file.
    	Capturing raw events...
    	1317 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/init-in-streams-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/init-in-streams-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 1277 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/init-in-streams-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       82.3       194755553           3      64918517.7           25324       194684078  cudaMallocManaged                                                               
        8.4        19970065           4       4992516.3           10379        19024135  cudaMemPrefetchAsync                                                            
        7.8        18524673           3       6174891.0         5163101         8064536  cudaFree                                                                        
        1.4         3254972           1       3254972.0         3254972         3254972  cudaDeviceSynchronize                                                           
        0.0           69546           4         17386.5           10104           37114  cudaLaunchKernel                                                                
        0.0           50008           3         16669.3            4577           40116  cudaStreamDestroy                                                               
        0.0           21581           3          7193.7            3479           13491  cudaStreamCreate                                                                
    
    
    
    
    Generating CUDA Kernel Statistics...
    
    Generating CUDA Memory Operation Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       55.0          609557           3        203185.7          190140          223516  initWith                                                                        
       45.0          498902           1        498902.0          498902          498902  addVectorsInto                                                                  
    
    
    CUDA Memory Operation Statistics (nanoseconds)
    
    Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
      100.0        10704416          64        167256.5          166240          184704  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    CUDA Memory Operation Statistics (KiB)
    
                Total      Operations            Average            Minimum            Maximum  Name                                                                            
    -----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
             131072.0              64             2048.0           2048.000             2048.0  [CUDA Unified Memory memcpy DtoH]                                               
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       44.8       432195683          31      13941796.2            2706       100177491  poll                                                                            
       42.9       413874407          26      15918246.4           24210       100125279  sem_timedwait                                                                   
        9.9        95085374         584        162817.4            1000        18998128  ioctl                                                                           
        2.2        20895947          88        237453.9            1653         8015404  mmap                                                                            
        0.1          644782          96          6716.5            1606           14705  fopen                                                                           
        0.1          517061          78          6629.0            3857           13206  open64                                                                          
        0.0          358825           3        119608.3           37295          222536  sem_wait                                                                        
        0.0          250700          89          2816.9            1562            3825  fclose                                                                          
        0.0          189905           5         37981.0           33553           48588  pthread_create                                                                  
        0.0          138383          14          9884.5            1815           14629  write                                                                           
        0.0           92578           3         30859.3           24284           41279  fgets                                                                           
        0.0           90331          81          1115.2            1003            4429  fcntl                                                                           
        0.0           38761          12          3230.1            2050            4791  munmap                                                                          
        0.0           36253          16          2265.8            1387            3896  read                                                                            
        0.0           32692           5          6538.4            4479            9399  open                                                                            
        0.0           15994           3          5331.3            4522            5808  pipe2                                                                           
        0.0           11714           5          2342.8            1987            2703  mprotect                                                                        
        0.0           11536           2          5768.0            4971            6565  socket                                                                          
        0.0            8918           3          2972.7            1842            4360  fread                                                                           
        0.0            7879           1          7879.0            7879            7879  connect                                                                         
        0.0            6067           1          6067.0            6067            6067  pthread_cond_broadcast                                                          
        0.0            2647           1          2647.0            2647            2647  bind                                                                            
        0.0            2007           1          2007.0            2007            2007  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


---
## Summary

At this point in the lab you are able to:

- Use the **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
- Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
- Utilize CUDA streams for concurrent kernel execution in accelerated applications.

At this point in time you have a wealth of fundamental tools and techniques for accelerating CPU-only applications, and for then optimizing those accelerated applications. In the final exercise, you will have a chance to apply everything that you've learned to accelerate an [n-body](https://en.wikipedia.org/wiki/N-body_problem) simulator, which predicts the individual motions of a group of objects interacting with each other gravitationally.

---
## Final Exercise: Accelerate and Optimize an N-Body Simulator

An [n-body](https://en.wikipedia.org/wiki/N-body_problem) simulator predicts the individual motions of a group of objects interacting with each other gravitationally. [01-nbody.cu](../edit/09-nbody/01-nbody.cu) contains a simple, though working, n-body simulator for bodies moving through 3 dimensional space. The application can be passed a command line argument to affect how many bodies are in the system.

In its current CPU-only form, working on 4096 bodies, this application is able to calculate about 30 million interactions between bodies in the system per second. Your task is to:

- GPU accelerate the program, retaining the correctness of the simulation
- Work iteratively to optimize the simulator so that it calculates over 30 billion interactions per second while working on 4096 bodies `(2<<11)`
- Work iteratively to optimize the simulator so that it calculates over 325 billion interactions per second while working on ~65,000 bodies `(2<<15)`

**After completing this, go back in the browser page you used to open this notebook and click the Assess button. If you have retained the accuracy of the application and accelerated it to the specifications above, you will receive a certification for your competency in the _Fundamentals of Accelerating Applications with CUDA C/C++_.**

### Considerations to Guide Your Work

Here are some things to consider before beginning your work:

- Especially for your first refactors, the logic of the application, the `bodyForce` function in particular, can and should remain largely unchanged: focus on accelerating it as easily as possible.
- You will not be able to accelerate the `randomizeBodies` function since it is using the `rand` function, which is not available on GPU devices. `randomizeBodies` is a host function. Do not touch it at all.
- The codebase contains a for-loop inside `main` for integrating the interbody forces calculated by `bodyForce` into the positions of the bodies in the system. This integration both needs to occur after `bodyForce` runs, and, needs to complete before the next call to `bodyForce`. Keep this in mind when choosing how and where to parallelize.
- Use a **profile driven** and iterative approach.
- You are not required to add error handling to your code, but you might find it helpful, as you are responsible for your code working correctly.

Have Fun!


```python
!nvcc -o nbody 09-nbody/01-nbody.cu
```


```python
!./nbody 11 # This argument is passed as `N` in the formula `2<<N`, to determine the number of bodies in the system
# Simulator is calculating positions correctly. 4096 Bodies: average 0.036 Billion Interactions / second
# Simulator is calculating positions correctly. 4096 Bodies: average 336.217 Billion Interactions / second
```

    Simulator is calculating positions correctly.
    4096 Bodies: average 336.217 Billion Interactions / second


Don't forget that you can use the `-f` flag to force the overwrite of an existing report file, so that you do not need to keep multiple report files around during development.


```python
!nsys profile --stats=true -o nbody-report ./nbody
```

    
    **** collection configuration ****
    	output_filename = /dli/task/nbody-report
    	force-overwrite = false
    	stop-on-exit = true
    	export_sqlite = true
    	stats = true
    	capture-range = none
    	stop-on-range-end = false
    	Beta: ftrace events:
    	ftrace-keep-user-config = false
    	trace-GPU-context-switch = false
    	delay = 0 seconds
    	duration = 0 seconds
    	kill = signal number 15
    	inherit-environment = true
    	show-output = true
    	trace-fork-before-exec = false
    	sample_cpu = true
    	backtrace_method = LBR
    	wait = all
    	trace_cublas = false
    	trace_cuda = true
    	trace_cudnn = false
    	trace_nvtx = true
    	trace_mpi = false
    	trace_openacc = false
    	trace_vulkan = false
    	trace_opengl = true
    	trace_osrt = true
    	osrt-threshold = 0 nanoseconds
    	cudabacktrace = false
    	cudabacktrace-threshold = 0 nanoseconds
    	profile_processes = tree
    	application command = ./nbody
    	application arguments = 
    	application working directory = /dli/task
    	NVTX profiler range trigger = 
    	NVTX profiler domain trigger = 
    	environment variables:
    	Collecting data...
    The application process terminated. One or more process it created re-parented. Waiting for termination of re-parented processes. To modify this behavior, use the `--wait` option.
     Simulator is calculating positions correctly.
    4096 Bodies: average 321.403 Billion Interactions / second
    	Generating the /dli/task/nbody-report.qdstrm file.
    	Capturing raw events...
    	1197 total events collected.
    	Saving diagnostics...
    	Saving qdstrm file to disk...
    	Finished saving file.
    
    
    Importing the qdstrm file using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/QdstrmImporter.
    
    Importing...
    
    Importing [==================================================100%]
    Saving report to file "/dli/task/nbody-report.qdrep"
    Report file saved.
    Please discard the qdstrm file and use the qdrep file instead.
    
    Removed /dli/task/nbody-report.qdstrm as it was successfully imported.
    Please use the qdrep file instead.
    
    Exporting the qdrep file to SQLite database using /opt/nvidia/nsight-systems/2019.5.2/host-linux-x64/nsys-exporter.
    
    Exporting 1160 events:
    
    0%   10   20   30   40   50   60   70   80   90   100%
    |----|----|----|----|----|----|----|----|----|----|
    ***************************************************
    
    Exported successfully to
    /dli/task/nbody-report.sqlite
    
    Generating CUDA API Statistics...
    CUDA API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       99.3       174990162           1     174990162.0       174990162       174990162  cudaHostAlloc                                                                   
        0.3          499349           1        499349.0          499349          499349  cudaFreeHost                                                                    
        0.2          355944           2        177972.0           41904          314040  cudaMemcpy                                                                      
        0.1          179564          20          8978.2            7034           32759  cudaLaunchKernel                                                                
        0.1          153470           1        153470.0          153470          153470  cudaMalloc                                                                      
        0.1          126832           1        126832.0          126832          126832  cudaFree                                                                        
    
    
    
    
    Generating CUDA Kernel Statistics...
    
    Generating CUDA Memory Operation Statistics...
    CUDA Kernel Statistics (nanoseconds)
    
    Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       96.7          432465          10         43246.5           42373           47621  bodyForce                                                                       
        3.3           14721          10          1472.1            1408            1856  integrate_position                                                              
    
    
    CUDA Memory Operation Statistics (nanoseconds)
    
    Time(%)      Total Time  Operations         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       58.8           12257           1         12257.0           12257           12257  [CUDA memcpy HtoD]                                                              
       41.2            8577           1          8577.0            8577            8577  [CUDA memcpy DtoH]                                                              
    
    
    CUDA Memory Operation Statistics (KiB)
    
                Total      Operations            Average            Minimum            Maximum  Name                                                                            
    -----------------  --------------  -----------------  -----------------  -----------------  --------------------------------------------------------------------------------
                 96.0               1               96.0             96.000               96.0  [CUDA memcpy HtoD]                                                              
                 96.0               1               96.0             96.000               96.0  [CUDA memcpy DtoH]                                                              
    
    
    
    
    Generating Operating System Runtime API Statistics...
    Operating System Runtime API Statistics (nanoseconds)
    
    Time(%)      Total Time       Calls         Average         Minimum         Maximum  Name                                                                            
    -------  --------------  ----------  --------------  --------------  --------------  --------------------------------------------------------------------------------
       53.7       208616278          12      17384689.8           10024       100114978  sem_timedwait                                                                   
       25.8       100313101          11       9119372.8           39265        30048472  poll                                                                            
       19.2        74722663         567        131786.0            1025        15528060  ioctl                                                                           
        0.7         2836025          80         35450.3            1659          779666  mmap                                                                            
        0.2          662464          97          6829.5            1777           23938  fopen                                                                           
        0.1          517491          79          6550.5            3971           14221  open64                                                                          
        0.1          254623          90          2829.1            1560            4172  fclose                                                                          
        0.0          145564           4         36391.0           32280           40879  pthread_create                                                                  
        0.0          130868          10         13086.8            7354           33165  write                                                                           
        0.0           92251           3         30750.3           24025           41820  fgets                                                                           
        0.0           89922          80          1124.0            1010            4557  fcntl                                                                           
        0.0           33976           5          6795.2            4366            9660  open                                                                            
        0.0           26826          12          2235.5            1403            3776  read                                                                            
        0.0           19622           6          3270.3            2282            4657  munmap                                                                          
        0.0           16740           4          4185.0            2774            6046  fread                                                                           
        0.0           16318           3          5439.3            4275            6253  pipe2                                                                           
        0.0           12929           2          6464.5            5523            7406  socket                                                                          
        0.0            8851           4          2212.8            2026            2338  mprotect                                                                        
        0.0            8189           1          8189.0            8189            8189  connect                                                                         
        0.0            6096           1          6096.0            6096            6096  pthread_cond_broadcast                                                          
        0.0            2762           1          2762.0            2762            2762  bind                                                                            
        0.0            2411           1          2411.0            2411            2411  listen                                                                          
    
    
    
    
    Generating NVTX Push-Pop Range Statistics...
    NVTX Push-Pop Range Statistics (nanoseconds)
    
    
    
    


## Advanced Content

The following sections, for those of you with time and interest, introduce more intermediate techniques involving some manual device memory management, and using non-default streams to overlap kernel execution and memory copies.

After learning about each of the techniques below, try to further optimize your nbody simulation using these techniques.

---
## Manual Device Memory Allocation and Copying

While `cudaMallocManaged` and `cudaMemPrefetchAsync` are performant, and greatly simplify memory migration, sometimes it can be worth it to use more manual methods for memory allocation. This is particularly true when it is known that data will only be accessed on the device or host, and the cost of migrating data can be reclaimed in exchange for the fact that no automatic on-demand migration is needed.

Additionally, using manual device memory management can allow for the use of non-default streams for overlapping data transfers with computational work. In this section you will learn some basic manual device memory allocation and copy techniques, before extending these techniques to overlap data copies with computational work. 

Here are some CUDA commands for manual device memory management:

- `cudaMalloc` will allocate memory directly to the active GPU. This prevents all GPU page faults. In exchange, the pointer it returns is not available for access by host code.
- `cudaMallocHost` will allocate memory directly to the CPU. It also "pins" the memory, or page locks it, which will allow for asynchronous copying of the memory to and from a GPU. Too much pinned memory can interfere with CPU performance, so use it only with intention. Pinned memory should be freed with `cudaFreeHost`.
- `cudaMemcpy` can copy (not transfer) memory, either from host to device or from device to host.

### Manual Device Memory Management Example

Here is a snippet of code that demonstrates the use of the above CUDA API calls.

```cpp
int *host_a, *device_a;        // Define host-specific and device-specific arrays.
cudaMalloc(&device_a, size);   // `device_a` is immediately available on the GPU.
cudaMallocHost(&host_a, size); // `host_a` is immediately available on CPU, and is page-locked, or pinned.

initializeOnHost(host_a, N);   // No CPU page faulting since memory is already allocated on the host.

// `cudaMemcpy` takes the destination, source, size, and a CUDA-provided variable for the direction of the copy.
cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

kernel<<<blocks, threads, 0, someStream>>>(device_a, N);

// `cudaMemcpy` can also copy data from device to host.
cudaMemcpy(host_a, device_a, size, cudaMemcpyDeviceToHost);

verifyOnHost(host_a, N);

cudaFree(device_a);
cudaFreeHost(host_a);          // Free pinned memory like this.
```

### Exercise: Manually Allocate Host and Device Memory

The most recent iteration of the vector addition application, [01-stream-init-solution](../edit/06-stream-init/solutions/01-stream-init-solution.cu), is using `cudaMallocManaged` to allocate managed memory first used on the device by the initialization kernels, then on the device by the vector add kernel, and then by the host, where the memory is automatically transfered, for verification. This is a sensible approach, but it is worth experimenting with some manual device memory allocation and copying to observe its impact on the application's performance.

Refactor the [01-stream-init-solution](../edit/06-stream-init/solutions/01-stream-init-solution.cu) application to **not** use `cudaMallocManaged`. In order to do this you will need to do the following:

- Replace calls to `cudaMallocManaged` with `cudaMalloc`.
- Create an additional vector that will be used for verification on the host. This is required since the memory allocated with `cudaMalloc` is not available to the host. Allocate this host vector with `cudaMallocHost`.
- After the `addVectorsInto` kernel completes, use `cudaMemcpy` to copy the vector with the addition results, into the host vector you created with `cudaMallocHost`.
- Use `cudaFreeHost` to free the memory allocated with `cudaMallocHost`.

Refer to [the solution](../edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu) if you get stuck.


```python
!nvcc -o vector-add-manual-alloc 06-stream-init/solutions/01-stream-init-solution.cu -run
```

    Success! All values calculated correctly.


After completing the refactor, open a report in Nsight Systems, and use the timeline to do the following:

- Notice that there is no longer a *Unified Memory* section of the timeline.
- Comparing this timeline to that of the previous refactor, compare the runtimes of `cudaMalloc` in the current application vs. `cudaMallocManaged` in the previous.
- Notice how in the current application, work on the initialization kernels does not start until a later time than it did in the previous iteration. Examination of the timeline will show the difference is the time taken by `cudaMallocHost`. This clearly points out the difference between memory transfers, and memory copies. When copying memory, as you are doing presently, the data will exist in 2 different places in the system. In the current case, the allocation of the 4th host-only vector incurs a small cost in performance, compared to only allocating 3 vectors in the previous iteration.

---
## Using Streams to Overlap Data Transfers and Code Execution

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.


```python
%%HTML

<div align="center"><iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQdHDR62S4hhvq02CZreC_Hvb9y89_IRIKtCQQ-eMItim744eRHOK6Gead5P_EaPj66Z3_NS0hlTRuh/embed?start=false&loop=false&delayms=3000" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>
```



<div align="center"><iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQdHDR62S4hhvq02CZreC_Hvb9y89_IRIKtCQQ-eMItim744eRHOK6Gead5P_EaPj66Z3_NS0hlTRuh/embed?start=false&loop=false&delayms=3000" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>



In addition to `cudaMemcpy` is `cudaMemcpyAsync` which can asynchronously copy memory either from host to device or from device to host as long as the host memory is pinned, which can be done by allocating it with `cudaMallocHost`.

Similar to kernel execution, `cudaMemcpyAsync` is only asynchronous by default with respect to the host. It executes, by default, in the default stream and therefore is a blocking operation with regard to other CUDA operations occuring on the GPU. The `cudaMemcpyAsync` function, however, takes as an optional 5th argument, a non-default stream. By passing it a non-default stream, the memory transfer can be concurrent to other CUDA operations occuring in other non-default streams.

A common and useful pattern is to use a combination of pinned host memory, asynchronous memory copies in non-default streams, and kernel executions in non-default streams, to overlap memory transfers with kernel execution.

In the following example, rather than wait for the entire memory copy to complete before beginning work on the kernel, segments of the required data are copied and worked on, with each copy/work segment running in its own non-default stream. Using this technique, work on parts of the data can begin while memory transfers for later segments occur concurrently. Extra care must be taken when using this technique to calculate segment-specific values for the number of operations, and the offset location inside arrays, as shown here:

```cpp
int N = 2<<24;
int size = N * sizeof(int);

int *host_array;
int *device_array;

cudaMallocHost(&host_array, size);               // Pinned host memory allocation.
cudaMalloc(&device_array, size);                 // Allocation directly on the active GPU device.

initializeData(host_array, N);                   // Assume this application needs to initialize on the host.

const int numberOfSegments = 4;                  // This example demonstrates slicing the work into 4 segments.
int segmentN = N / numberOfSegments;             // A value for a segment's worth of `N` is needed.
size_t segmentSize = size / numberOfSegments;    // A value for a segment's worth of `size` is needed.

// For each of the 4 segments...
for (int i = 0; i < numberOfSegments; ++i)
{
  // Calculate the index where this particular segment should operate within the larger arrays.
  segmentOffset = i * segmentN;

  // Create a stream for this segment's worth of copy and work.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Asynchronously copy segment's worth of pinned host memory to device over non-default stream.
  cudaMemcpyAsync(&device_array[segmentOffset],  // Take care to access correct location in array.
                  &host_array[segmentOffset],    // Take care to access correct location in array.
                  segmentSize,                   // Only copy a segment's worth of memory.
                  cudaMemcpyHostToDevice,
                  stream);                       // Provide optional argument for non-default stream.
                  
  // Execute segment's worth of work over same non-default stream as memory copy.
  kernel<<<number_of_blocks, threads_per_block, 0, stream>>>(&device_array[segmentOffset], segmentN);
  
  // `cudaStreamDestroy` will return immediately (is non-blocking), but will not actually destroy stream until
  // all stream operations are complete.
  cudaStreamDestroy(stream);
}
```

### Exercise: Overlap Kernel Execution and Memory Copy Back to Host

The most recent iteration of the vector addition application, [01-manual-malloc-solution.cu](../edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu), is currently performing all of its vector addition work on the GPU before copying the memory back to the host for verification.

Refactor [01-manual-malloc-solution.cu](../edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu) to perform the vector addition in 4 segments, in non-default streams, so that asynchronous memory copies can begin before waiting for all vector addition work to complete. Refer to [the solution](../edit/08-overlap-xfer/solutions/01-overlap-xfer-solution.cu) if you get stuck.


```python
!nvcc -o vector-add-manual-alloc 07-manual-malloc/solutions/01-manual-malloc-solution.cu -run
```

    Success! All values calculated correctly.


After completing the refactor, open a report in Nsight Systems, and use the timeline to do the following:

- Note when the device to host memory transfers begin, is it before or after all kernel work has completed?
- Notice that the 4 memory copy segments themselves do not overlap. Even in separate non-default streams, only one memory transfer in a given direction (DtoH here) at a time can occur simultaneously. The performance gains here are in the ability to start the transfers earlier than otherwise, and it is not hard to imagine in an application where a less trivial amount of work was being done compared to a simple addition operation, that the memory copies would not only start earlier, but also overlap with kernel execution.

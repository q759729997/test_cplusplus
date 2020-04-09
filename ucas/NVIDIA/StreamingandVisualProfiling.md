# **Asynchronous Streaming, and Visual Profiling with CUDA C/C++**

![CUDA](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/CUDA_Logo.jpg)CUDA

The CUDA tookit ships with the **Nsight Systems**, a powerful GUI application to support the development of accelerated CUDA applications. Nsight Systems generates a graphical timeline of an accelerated application, with detailed information about CUDA API calls, kernel execution, memory activity, and the use of **CUDA streams**.

CUDA tookit附带了**Nsight Systems**，这是一个强大的GUI应用程序，支持加速CUDA应用程序的开发。Nsight Systems生成一个加速应用程序的图形时间线，其中包含有关CUDA API调用、内核执行、内存活动和使用**CUDA streams**的详细信息。

In this lab, you will be using the Nsight Systems timeline to guide you in optimizing accelerated applications. Additionally, you will learn some intermediate CUDA programming techniques to support your work: **unmanaged memory allocation and migration**; **pinning**, or **page-locking** host memory; and **non-default concurrent CUDA streams**.

在这个实验室中，您将使用Nsight系统时间线来指导您优化加速应用程序。此外，您还将学习一些中间CUDA编程技术来支持您的工作：**非托管内存分配和迁移**；**固定**，或**页面锁定**主机内存；以及**非默认并发CUDA流**。

At the end of this lab, you will be presented with an assessment, to accelerate and optimize a simple n-body particle simulator, which will allow you to demonstrate the skills you have developed during this course. Those of you who are able to accelerate the simulator while maintaining its correctness, will be granted a certification as proof of your competency.

在本实验室结束时，将向您展示一个评估，以加速和优化一个简单的n体粒子模拟器，这将允许您演示您在本课程中开发的技能。你们中那些能够在保持模拟器正确性的同时加速模拟器的人，将被授予证书作为你们能力的证明。

------

## Prerequisites 先决条件

To get the most out of this lab you should already be able to:

- Write, compile, and run C/C++ programs that both call CPU functions and launch GPU kernels.
- Control parallel thread hierarchy using execution configuration.
- Refactor serial loops to execute their iterations in parallel on a GPU.
- Allocate and free CUDA Unified Memory.
- Understand the behavior of Unified Memory with regard to page faulting and data migrations.
- Use asynchronous memory prefetching to reduce page faults and data migrations.

为了充分利用这个实验室，您应该已经能够：

- 编写、编译和运行C/C++程序，这些程序既调用CPU功能又启动GPU内核。

- 使用执行配置控制并行线程层次结构。

- 重构串行循环以在GPU上并行执行它们的迭代。

- 分配和释放CUDA统一内存。

- 了解统一内存在页错误和数据迁移方面的行为。

- 使用异步内存预取来减少页面错误和数据迁移。

## Objectives 目标

By the time you complete this lab you will be able to:

- Use **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
- Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
- Utilize CUDA streams for concurrent kernel execution in accelerated applications.
- (**Optional Advanced Content**) Use manual device memory allocation, including allocating pinned memory, in order to asynchronously transfer data in concurrent CUDA streams.

完成本实验后，您将能够：

- 使用**Nsight Systems**直观地描述GPU加速CUDA应用程序的时间线。

- 使用Nsight系统识别并利用GPU加速CUDA应用程序中的优化机会。

- 在加速应用程序中利用CUDA流进行并发内核执行。

- （**可选高级内容**）使用手动设备内存分配，包括分配固定内存，以便在并发CUDA流中异步传输数据。

------

## Running Nsight Systems 运行Nsight系统

For this interactive lab environment, we have set up a remote desktop you can access from your browser, where you will be able to launch and use Nsight Systems.

对于这个交互式的实验室环境，我们已经设置了一个远程桌面，您可以从浏览器访问，在那里您可以启动和使用Nsight系统。

You will begin by creating a report file for an already-existing vector addition program, after which you will be walked through a series of steps to open this report file in Nsight Systems, and to make the visual experience nice.

首先，您将为一个已经存在的矢量添加程序创建一个报告文件，然后，您将经历一系列步骤，以便在Nsight系统中打开此报告文件，并使视觉体验更美好。

### Generate Report File

[`01-vector-add.cu`](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/01-vector-add/01-vector-add.cu) (<-------- click on these links to source files to edit them in the browser) contains a working, accelerated, vector addition application. Use the code execution cell directly below (you can execute it, and any of the code execution cells in this lab by `CTRL` + clicking it) to compile and run it. You should see a message printed that indicates it was successful.

（单击这些指向源文件的链接以在浏览器中编辑它们）包含一个工作的、加速的矢量添加应用程序。使用下面的代码执行单元（您可以执行它，也可以通过“CTRL”+单击它来执行这个实验室中的任何代码执行单元）编译并运行它。您应该会看到一条打印的消息，指示它已成功。

In [ ]:

```
!nvcc -o vector-add-no-prefetch 01-vector-add/01-vector-add.cu -run
```

Next, use `nsys profile --stats=true` to create a report file that you will be able to open in the Nsight Systems visual profiler. Here we use the `-o` flag to give the report file a memorable name:

接下来，使用“nsys profile--stats=true”创建一个报告文件，您可以在Nsight Systems visual profiler中打开该文件。在这里，我们使用“-o”标志为报告文件命名：

In [ ]:

```
!nsys profile --stats=true -o vector-add-no-prefetch-report ./vector-add-no-prefetch
```

### Open the Remote Desktop 打开远程桌面

Run the next cell to generate a link to a remote desktop then read the instructions that follow.

运行下一个单元格以生成指向远程桌面的链接，然后阅读下面的说明。

In [ ]:

```
%%js
var url = window.location.hostname + ':5901';
element.append('<a href="'+url+'">Open Remote Desktop</a>')
```

After clicking the *Connect* button you will be asked for a password, which is `nvidia`.

单击*连接*按钮后，系统将要求您输入密码，即“nvidia”。

### Open the Remote Desktop Terminal Application 打开远程桌面终端应用程序

Next, click on the icon for the terminal application, found at the bottom of the screen of the remote desktop:

接下来，单击位于远程桌面屏幕底部的终端应用程序图标：

![terminal](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/terminal.png)

### Open Nsight Systems 打开Nsight系统

To open Nsight Systems, enter and run the `nsight-sys` command from the now-open terminal:

要打开Nsight系统，请从现在打开的终端输入并运行“Nsight sys”命令：

![open nsight](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/open-nsight.png)

### Enable Usage Reporting 启用使用情况报告

When prompted, click "Yes" to enable usage reporting:

出现提示时，单击“是”以启用使用情况报告：

![enable usage](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/enable_usage.png)

### Open the Report File 打开报告文件

Open this report file by visiting *File* -> *Open* from the Nsight Systems menu, then go to the path `/root/Desktop/reports` and select `vector-add-no-prefetch-report.qdrep`. All the reports you generate in this lab will be in this `root/Desktop/reports` directory:

通过访问Nsight Systems菜单中的*file*>*Open*打开此报表文件，然后转到路径`/root/Desktop/reports`并选择“vector add no prefetch report.qdrep”。您在此实验室中生成的所有报告都将位于此“root/Desktop/reports”目录中：

![open-report](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/open-report.png)

### Ignore Warnings/Errors 忽略警告/错误

You can close and ignore any warnings or errors you see, which are just a result of our particular remote desktop environment:

您可以关闭并忽略您看到的任何警告或错误，这些警告或错误只是特定远程桌面环境的结果：

![ignore errors](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/ignore-error.png)

### Make More Room for the Timelines 为时间线腾出更多空间

To make your experience nicer, full-screen the profiler, close the *Project Explorer* and hide the *Events View*:

要使您的体验更好，请全屏显示探查器，关闭*项目资源管理器*并隐藏*事件视图*：

![make nice](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/make-nice.png)

Your screen should now look like this:

您的屏幕现在应该如下所示：

![now nice](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/now-nice.png)

### Expand the CUDA Unified Memory Timelines 扩展CUDA统一内存时间表

Next, expand the *CUDA* -> *Unified memory* and *Context* timelines, and close the *OS runtime libraries* timelines:

接下来，展开*CUDA*>*统一内存*和*Context*时间线，并关闭*OS运行库*时间线：

![open memory](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/open-memory.png)

### Observe Many Memory Transfers 观察多内存传输

From a glance you can see that your application is taking about 1 second to run, and that also, during the time when the `addVectorsInto` kernel is running, that there is a lot of UM memory activity:

从一眼就能看出，应用程序的运行时间大约为1秒，而且在“addVectorsInto”内核运行期间，存在大量的UM内存活动：

![memory and kernel](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/memory-and-kernel.png)

Zoom into the memory timelines to see more clearly all the small memory transfers being caused by the on-demand memory page faults. A couple tips:

1. You can zoom in and out at any point of the timeline by holding `Ctrl` while scrolling your mouse/trackpad
2. You can zoom into any section by click + dragging a rectangle around it, and then selecting *Zoom in*

Here's an example of zooming in to see the many small memory transfers:

放大内存时间线，以便更清楚地看到由按需内存页错误导致的所有小内存传输。几点建议：

1. 在滚动鼠标/轨迹板时按住“Ctrl”，可以在时间轴的任何点上放大或缩小

2. 您可以通过单击并在其周围拖动矩形，然后选择*放大来放大任何部分*

下面是放大以查看许多小内存传输的示例：

![many transfers](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/many-transfers.png)

------

## Comparing Code Refactors Iteratively with Nsight Systems 代码重构与Nsight系统的迭代比较

Now that you have Nsight Systems up and running and are comfortable moving around the timelines, you will be profiling a series of programs that were iteratively improved using techniques already familiar to you. Each time you profile, information in the timeline will give information supporting how you should next modify your code. Doing this will further increase your understanding of how various CUDA programming techniques affect application performance.

现在，您已经启动并运行了Nsight系统，并且可以轻松地在时间线上移动，您将分析一系列使用您已经熟悉的技术进行迭代改进的程序。每次分析时，时间线中的信息都会提供支持您下一步如何修改代码的信息。这样做将进一步加深您对各种CUDA编程技术如何影响应用程序性能的理解。

### Exercise: Compare the Timelines of Prefetching vs. Non-Prefetching 练习：比较预取与非预取的时间线

[`01-vector-add-prefetch-solution.cu`](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/01-vector-add/solutions/01-vector-add-prefetch-solution.cu) refactors the vector addition application from above so that the 3 vectors needed by its `addVectorsInto` kernel are asynchronously prefetched to the active GPU device prior to launching the kernel (using [`cudaMemPrefetchAsync`](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8dc9199943d421bc8bc7f473df12e42)). Open the source code and identify where in the application these changes were made.

从上面重构向量添加应用程序，以便在启动“addVectorsInto”内核之前，将其所需的3个向量异步预取到活动GPU设备内核（使用[`cudaMemPrefetchAsync`]）。打开源代码并确定这些更改在应用程序中的位置。

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.

在检查更改之后，使用下面的代码执行单元编译并运行重构的应用程序。你应该看到它的成功信息被打印出来。

In [ ]:

```
!nvcc -o vector-add-prefetch 01-vector-add/solutions/01-vector-add-prefetch-solution.cu -run
```

Now create a report file for this version of the application:

现在为该版本的应用程序创建一个报告文件：

In [ ]:

```
!nsys profile --stats=true -o vector-add-prefetch-report ./vector-add-prefetch
```

Open the report in Nsight Systems, leaving the previous report open for comparison.

- How does the execution time compare to that of the `addVectorsInto` kernel prior to adding asynchronous prefetching?
- Locate `cudaMemPrefetchAsync` in the *CUDA API* section of the timeline.
- How have the memory transfers changed?

在Nsight系统中打开报告，保留上一个报告以供比较。

- 在添加异步预取之前，执行时间与“addVectorsInto”内核的执行时间相比如何？

- 在时间线的*CUDA API*部分找到“cudaMemPrefetchAsync”。

- 内存传输是如何改变的？

### Exercise: Profile Refactor with Launch Init in Kernel 练习：在内核中使用Launch Init进行概要文件重构

In the previous iteration of the vector addition application, the vector data is being initialized on the CPU, and therefore needs to be migrated to the GPU before the `addVectorsInto` kernel can operate on it.

在向量加法应用程序的上一次迭代中，向量数据正在CPU上初始化，因此需要在“addVectorsInto”内核可以对其进行操作之前迁移到GPU。

The next iteration of the application, [01-init-kernel-solution.cu](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/02-init-kernel/solutions/01-init-kernel-solution.cu), the application has been refactored to initialize the data in parallel on the GPU.

在应用程序的下一次迭代中，[01 init kernel solution.cu]，应用程序已被重构以在GPU上并行初始化数据。

Since the initialization now takes place on the GPU, prefetching has been done prior to initialization, rather than prior to the vector addition work. Review the source code to identify where these changes have been made.

由于现在在GPU上进行初始化，所以预取是在初始化之前完成的，而不是在矢量添加工作之前。检查源代码以确定这些更改是在哪里进行的。

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.

在检查更改之后，使用下面的代码执行单元编译并运行重构的应用程序。你应该看到它的成功信息被打印出来。

In [ ]:

```
!nvcc -o init-kernel 02-init-kernel/solutions/01-init-kernel-solution.cu -run
```

Now create a report file for this version of the application:

现在为该版本的应用程序创建一个报告文件：

In [ ]:

```
!nsys profile --stats=true -o init-kernel-report ./init-kernel
```

Open the new report file in Nsight Systems and do the following:

- Compare the application and `addVectorsInto` runtimes to the previous version of the application, how did they change?
- Look at the *Kernels* section of the timeline. Which of the two kernels (`addVectorsInto` and the initialization kernel) is taking up the majority of the time on the GPU?
- Which of the following does your application contain?
  - Data Migration (HtoD)
  - Data Migration (DtoH)

在Nsight系统中打开新报告文件并执行以下操作：

- 将应用程序和“addVectorsInto”运行时与以前版本的应用程序进行比较，它们是如何更改的？

- 看看时间线的“内核”部分。两个内核中的哪一个（addVectorsInto和初始化内核）占据了GPU上的大部分时间？

- 你的申请包含下列哪一项？

  - 数据迁移（HtoD）

  - 数据迁移（DtoH）

### Exercise: Profile Refactor with Asynchronous Prefetch Back to the Host 练习：异步预取回主机的概要文件重构

Currently, the vector addition application verifies the work of the vector addition kernel on the host. The next refactor of the application, [01-prefetch-check-solution.cu](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/04-prefetch-check/solutions/01-prefetch-check-solution.cu), asynchronously prefetches the data back to the host for verification.

目前，向量加法应用程序在主机上验证向量加法核的工作。应用程序的下一个重构，[01 prefetch check solution.cu]异步地将数据预取回主机进行验证。

After reviewing the changes, compile and run the refactored application using the code execution cell directly below. You should see its success message printed.

在检查更改之后，使用下面的代码执行单元编译并运行重构的应用程序。你应该看到它的成功信息被打印出来。

In [ ]:

```
!nvcc -o prefetch-to-host 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

Now create a report file for this version of the application:

现在为该版本的应用程序创建一个报告文件：

In [ ]:

```
!nsys profile --stats=true -o prefetch-to-host-report ./prefetch-to-host
```

Open this report file in Nsight Systems, and do the following:

- Use the *Unified Memory* section of the timeline to compare and contrast the *Data Migration (DtoH)* events before and after adding prefetching back to the CPU.

在Nsight系统中打开此报表文件，并执行以下操作：

- 使用时间线的*Unified Memory*部分比较和对比添加预取回CPU之前和之后的*Data Migration（DtoH）*事件。

------

## Concurrent CUDA Streams 竞争对手CUDA Streams

You are now going to learn about a new concept, **CUDA Streams**. After an introduction to them, you will return to using Nsight Systems to better evaluate their impact on your application's performance.

现在您将学习一个新概念，**CUDA Streams**。在介绍了它们之后，您将返回到使用Nsight系统来更好地评估它们对应用程序性能的影响。

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.

以下幻灯片以高层次直观地展示了即将到来的材料。单击幻灯片，然后转到下面各节中对其主题的更详细介绍。

In [ ]:

```
%%HTML

<div align="center"><iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRVgzpDzp5fWAu-Zpuyr09rmIqE4FTFESjajhfZSnY7yVvPgZUDxECAPSdLko5DZNTGEN7uA79Hfovd/embed?start=false&loop=false&delayms=3000" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>
```

In CUDA programming, a **stream** is a series of commands that execute in order. In CUDA applications, kernel execution, as well as some memory transfers, occur within CUDA streams. Up until this point in time, you have not been interacting explicitly with CUDA streams, but in fact, your CUDA code has been executing its kernels inside of a stream called *the default stream*.

在CUDA编程中，**stream**是一系列按顺序执行的命令。在CUDA应用程序中，内核执行以及一些内存传输发生在CUDA流中。到目前为止，您还没有显式地与CUDA流交互，但事实上，您的CUDA代码一直在名为*the default stream*的流中执行其内核。

CUDA programmers can create and utilize non-default CUDA streams in addition to the default stream, and in doing so, perform multiple operations, such as executing multiple kernels, concurrently, in different streams. Using multiple streams can add an additional layer of parallelization to your accelerated applications, and offers many more opportunities for application optimization.

CUDA程序员可以在默认流之外创建和使用非默认的CUDA流，在这样做的过程中，可以执行多个操作，例如在不同的流中并发地执行多个内核。使用多个流可以为加速的应用程序添加额外的并行层，并为应用程序优化提供更多的机会。

### Rules Governing the Behavior of CUDA Streams 控制CUDA流行为的规则

There are a few rules, concerning the behavior of CUDA streams, that should be learned in order to utilize them effectively:

- Operations within a given stream occur in order.
- Operations in different non-default streams are not guaranteed to operate in any specific order relative to each other.
- The default stream is blocking and will both wait for all other streams to complete before running, and, will block other streams from running until it completes.

为了有效地利用CUDA流，需要学习一些关于CUDA流行为的规则：

- 给定流中的操作按顺序发生。

- 不同非默认流中的操作不能保证以任何特定的顺序彼此操作。

- 默认的流是阻塞的，在运行之前都将等待所有其他流完成，并且，将阻塞其他流直到它完成为止。

### Creating, Utilizing, and Destroying Non-Default CUDA Streams 创建、利用和销毁非默认CUDA流

The following code snippet demonstrates how to create, utilize, and destroy a non-default CUDA stream. You will note, that to launch a CUDA kernel in a non-default CUDA stream, the stream must be passed as the optional 4th argument of the execution configuration. Up until now you have only utilized the first 2 arguments of the execution configuration:

下面的代码片段演示如何创建、利用和销毁非默认CUDA流。您将注意到，要在非默认CUDA流中启动CUDA内核，该流必须作为执行配置的可选第4个参数传递。到目前为止，您只使用了执行配置的前两个参数：

```cpp
cudaStream_t stream;       // CUDA streams are of type `cudaStream_t`.
cudaStreamCreate(&stream); // Note that a pointer must be passed to `cudaCreateStream`.

someKernel<<<number_of_blocks, threads_per_block, 0, stream>>>(); // `stream` is passed as 4th EC argument.

cudaStreamDestroy(stream); // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
```

Outside the scope of this lab, but worth mentioning, is the optional 3rd argument of the execution configuration. This argument allows programmers to supply the number of bytes in **shared memory** (an advanced topic that will not be covered presently) to be dynamically allocated per block for this kernel launch. The default number of bytes allocated to shared memory per block is `0`, and for the remainder of the lab, you will be passing `0` as this value, in order to expose the 4th argument, which is of immediate interest:

在这个实验室的范围之外，但值得一提的是，执行配置的第三个可选参数。此参数允许程序员提供**共享内存**中的字节数（这是一个目前不涉及的高级主题），以便为此内核启动动态分配每个块。每个块分配给共享内存的默认字节数是'0'，对于实验室的其余部分，您将传递'0'作为此值，以便公开第4个参数，这是直接相关的：

### Exercise: Predict Default Stream Behavior 练习：预测默认流行为

The [01-print-numbers](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/05-stream-intro/01-print-numbers.cu) application has a very simple `printNumber` kernel which accepts an integer and prints it. The kernel is only being executed with a single thread inside a single block, however, it is being executed 5 times, using a for-loop, and passing each launch the number of the for-loop's iteration.

[01打印号码]应用程序有一个非常简单的“printNumber”内核，它接受一个整数并打印它。内核只在一个块中的一个线程中执行，但是，它执行了5次，使用for循环，并在每次启动时传递for循环的迭代次数。

Compile and run [01-print-numbers](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/05-stream-intro/01-print-numbers.cu) using the code execution block below. You should see the numbers `0` through `4` printed.

使用下面的代码执行块编译并运行[01 print numbers]。你应该看到数字“0”到“4”被打印出来。

In [ ]:

```
!nvcc -o print-numbers 05-stream-intro/01-print-numbers.cu -run
```

Knowing that by default kernels are executed in the default stream, would you expect that the 5 launches of the `print-numbers` program executed serially, or in parallel? You should be able to mention two features of the default stream to support your answer. Create a report file in the cell below and open it in Nsight Systems to confirm your answer.

知道默认情况下内核是在默认流中执行的，您希望“print numbers”程序的5次启动是串行执行还是并行执行？您应该能够提到默认流的两个特性来支持您的答案。在下面的单元格中创建一个报告文件，并在Nsight系统中打开它以确认您的答案。

In [ ]:

```
!nsys profile --stats=true -o print-numbers-report ./print-numbers
```

### Exercise: Implement Concurrent CUDA Streams 练习：实现并发CUDA流

Both because all 5 kernel launches occured in the same stream, you should not be surprised to have seen that the 5 kernels executed serially. Additionally you could make the case that because the default stream is blocking, each launch of the kernel would wait to complete before the next launch, and this is also true.

这两种情况都是因为所有5个内核启动都发生在同一个流中，所以您不必惊讶地看到5个内核是串行执行的。此外，您还可以假设，由于默认流被阻塞，内核的每次启动都将等待在下一次启动之前完成，这也是正确的。

Refactor [01-print-numbers](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/05-stream-intro/01-print-numbers.cu) so that each kernel launch occurs in its own non-default stream. Be sure to destroy the streams you create after they are no longer needed. Compile and run the refactored code with the code execution cell directly below. You should still see the numbers `0` through `4` printed, though not necessarily in ascending order. Refer to [the solution](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/05-stream-intro/solutions/01-print-numbers-solution.cu) if you get stuck.

重构[01 print numbers]，以便每个内核启动都在自己的非默认流中进行。确保在不再需要创建的流之后销毁它们。使用下面的代码执行单元编译并运行重构代码。您应该仍然可以看到从“0”到“4”的数字被打印出来，尽管不一定是按升序排列的。如果您遇到问题，请参考[解决方案]

In [ ]:

```
!nvcc -o print-numbers-in-streams 05-stream-intro/01-print-numbers.cu -run
```

Now that you are using 5 different non-default streams for each of the 5 kernel launches, do you expect that they will run serially or in parallel? In addition to what you now know about streams, take into account how trivial the `printNumber` kernel is, meaning, even if you predict parallel runs, will the speed at which one kernel will complete allow for complete overlap?

现在您为5个内核启动使用了5个不同的非默认流，您希望它们是串行运行还是并行运行？除了您现在对流的了解之外，还要考虑到“printNumber”内核是多么微不足道，这意味着，即使您预测并行运行，一个内核完成的速度是否允许完全重叠？

After hypothesizing, open a new report file in Nsight Systems to view its actual behavior. You should notice that now, there are additional rows in the *CUDA* section for each of the non-default streams you created:

假设之后，在Nsight系统中打开一个新的报告文件来查看它的实际行为。您应该注意到，现在，*CUDA*部分中为您创建的每个非默认流添加了一些行：

In [ ]:

```
!nsys profile --stats=true -o print-numbers-in-streams-report print-numbers-in-streams
```

![streams overlap](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/notebooks/images/streams-overlap.png)

### Exercise: Use Streams for Concurrent Data Initialization Kernels 练习：对并发数据初始化内核使用流

The vector addition application you have been working with, [01-prefetch-check-solution.cu](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/04-prefetch-check/solutions/01-prefetch-check-solution.cu), currently launches an initialization kernel 3 times - once each for each of the 3 vectors needing initialization for the `vectorAdd` kernel. Refactor it to launch each of the 3 initialization kernel launches in their own non-default stream. You should still be see the success message print when compiling and running with the code execution cell below. Refer to [the solution](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/06-stream-init/solutions/01-stream-init-solution.cu) if you get stuck.

您一直在使用的向量添加应用程序，[01 prefetch check solution.cu]当前会启动一个初始化内核3次，对于需要初始化“vectorAdd”内核的3个向量，每个都会启动一次。重构它以在3个初始化内核中的每一个在它们自己的非默认流中启动。在编译并使用下面的代码执行单元运行时，您应该仍然可以看到成功消息打印。如果遇到问题，请参阅[解决方案]

In [ ]:

```
!nvcc -o init-in-streams 04-prefetch-check/solutions/01-prefetch-check-solution.cu -run
```

Open a report in Nsight Systems to confirm that your 3 initialization kernel launches are running in their own non-default streams, with some degree of concurrent overlap.

在Nsight系统中打开一个报告，确认您的3个初始化内核启动是在它们自己的非默认流中运行的，并且有一定程度的并发重叠。

In [ ]:

```
!nsys profile --stats=true -o init-in-streams-report ./init-in-streams
```

------

## Summary 总结

At this point in the lab you are able to:

- Use the **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
- Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
- Utilize CUDA streams for concurrent kernel execution in accelerated applications.

在实验室的这一点上，您可以：

- 使用**Nsight Systems**直观地分析GPU加速CUDA应用程序的时间线。

- 使用Nsight系统识别并利用GPU加速CUDA应用程序中的优化机会。

- 在加速应用程序中利用CUDA流进行并发内核执行。

At this point in time you have a wealth of fundamental tools and techniques for accelerating CPU-only applications, and for then optimizing those accelerated applications. In the final exercise, you will have a chance to apply everything that you've learned to accelerate an [n-body](https://en.wikipedia.org/wiki/N-body_problem) simulator, which predicts the individual motions of a group of objects interacting with each other gravitationally.

在这个时间点上，您有大量的基本工具和技术来加速仅限CPU的应用程序，然后优化那些加速的应用程序。在最后一个练习中，你将有机会应用你所学的一切来加速一个[n-body]模拟器（https://en.wikipedia.org/wiki/n-body_problem），它可以预测一组物体在引力作用下相互作用的各个运动。

------

## Final Exercise: Accelerate and Optimize an N-Body Simulator 最后练习：加速和优化N体模拟器

An [n-body](https://en.wikipedia.org/wiki/N-body_problem) simulator predicts the individual motions of a group of objects interacting with each other gravitationally. [01-nbody.cu](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/09-nbody/01-nbody.cu) contains a simple, though working, n-body simulator for bodies moving through 3 dimensional space. The application can be passed a command line argument to affect how many bodies are in the system.

一个[n-body]模拟器（https://en.wikipedia.org/wiki/n-body_problem）预测一组物体在引力作用下相互作用的个体运动。[01 n body.cu]包含一个简单的、可工作的n体模拟器，用于在三维空间中移动的物体。可以向应用程序传递命令行参数，以影响系统中有多少实体。

In its current CPU-only form, working on 4096 bodies, this application is able to calculate about 30 million interactions between bodies in the system per second. Your task is to:

- GPU accelerate the program, retaining the correctness of the simulation
- Work iteratively to optimize the simulator so that it calculates over 30 billion interactions per second while working on 4096 bodies `(2<<11)`
- Work iteratively to optimize the simulator so that it calculates over 325 billion interactions per second while working on ~65,000 bodies `(2<<15)`

这个应用程序目前只使用CPU，处理4096个实体，每秒能够计算系统中实体之间的大约3000万次交互。你的任务是：

- GPU加速程序，保持仿真的正确性

- 以迭代方式优化模拟器，使其在处理4096个物体时每秒计算超过300亿次的相互作用`（2<<11）`

- 以迭代方式优化模拟器，使其在处理约65000个物体时每秒计算3250多亿次相互作用（2<<15）`

**After completing this, go back in the browser page you used to open this notebook and click the Assess button. If you have retained the accuracy of the application and accelerated it to the specifications above, you will receive a certification for your competency in the \*Fundamentals of Accelerating Applications with CUDA C/C++\*.**

完成此操作后，返回用于打开此笔记本的浏览器页，然后单击“评估”按钮。如果您保留了应用程序的准确性并将其加速到上述规范，您将获得一份证书，证明您在CUDA C/C++中加速应用程序的基本原理。

### Considerations to Guide Your Work 指导工作的注意事项

Here are some things to consider before beginning your work:

- Especially for your first refactors, the logic of the application, the `bodyForce` function in particular, can and should remain largely unchanged: focus on accelerating it as easily as possible.
- You will not be able to accelerate the `randomizeBodies` function since it is using the `rand` function, which is not available on GPU devices. `randomizeBodies` is a host function. Do not touch it at all.
- The codebase contains a for-loop inside `main` for integrating the interbody forces calculated by `bodyForce` into the positions of the bodies in the system. This integration both needs to occur after `bodyForce` runs, and, needs to complete before the next call to `bodyForce`. Keep this in mind when choosing how and where to parallelize.
- Use a **profile driven** and iterative approach.
- You are not required to add error handling to your code, but you might find it helpful, as you are responsible for your code working correctly.

在开始工作之前，需要考虑以下几点：

- 尤其是对于您的第一个重构，应用程序的逻辑，特别是“bodyForce”函数，可以而且应该基本保持不变：尽可能轻松地加速它。

- 您将无法加速“randomizeBodies”函数，因为它使用的是“rand”函数，而GPU设备上没有该函数。`randomizeBodies是一个宿主函数。别碰它。

- 代码库在“main”中包含一个for循环，用于将“bodyForce”计算出的身体间力集成到系统中身体的位置中。这种集成既需要在“bodyForce”运行之后进行，也需要在下次调用“bodyForce”之前完成。在选择并行方式和并行位置时，请记住这一点。

- 使用**配置文件驱动**和迭代方法。

- 您不需要向代码中添加错误处理，但您可能会发现它很有帮助，因为您要对代码的正常工作负责。

Have Fun!

In [ ]:

```
!nvcc -o nbody 09-nbody/01-nbody.cu
```

In [ ]:

```
!./nbody 11 # This argument is passed as `N` in the formula `2<<N`, to determine the number of bodies in the system
```

Don't forget that you can use the `-f` flag to force the overwrite of an existing report file, so that you do not need to keep multiple report files around during development.

不要忘记，可以使用“-f”标志强制覆盖现有报表文件，这样在开发过程中就不需要保留多个报表文件。

In [ ]:

```
!nsys profile --stats=true -o nbody-report ./nbody
```

## Advanced Content 高级内容

The following sections, for those of you with time and interest, introduce more intermediate techniques involving some manual device memory management, and using non-default streams to overlap kernel execution and memory copies.

下面的部分，对于那些有时间和兴趣的人，将介绍更多中间技术，包括一些手动设备内存管理，以及使用非默认流来重叠内核执行和内存副本。

After learning about each of the techniques below, try to further optimize your nbody simulation using these techniques.

在学习了下面的每一种技术之后，尝试使用这些技术进一步优化nbody模拟。

------

## Manual Device Memory Allocation and Copying 手动设备内存分配和复制

While `cudaMallocManaged` and `cudaMemPrefetchAsync` are performant, and greatly simplify memory migration, sometimes it can be worth it to use more manual methods for memory allocation. This is particularly true when it is known that data will only be accessed on the device or host, and the cost of migrating data can be reclaimed in exchange for the fact that no automatic on-demand migration is needed.

虽然“cudaMallocManaged”和“cudaMemPrefetchAsync”的性能良好，并且极大地简化了内存迁移，但有时使用更多的手动方法分配内存是值得的。当已知数据只在设备或主机上访问时，尤其如此，而且由于不需要自动按需迁移，因此可以回收迁移数据的成本。

Additionally, using manual device memory management can allow for the use of non-default streams for overlapping data transfers with computational work. In this section you will learn some basic manual device memory allocation and copy techniques, before extending these techniques to overlap data copies with computational work.

此外，使用手动设备内存管理可以允许使用非默认流来重叠数据传输和计算工作。在本节中，您将学习一些基本的手动设备内存分配和复制技术，然后再扩展这些技术以使数据复制与计算工作重叠。

Here are some CUDA commands for manual device memory management:

- `cudaMalloc` will allocate memory directly to the active GPU. This prevents all GPU page faults. In exchange, the pointer it returns is not available for access by host code.
- `cudaMallocHost` will allocate memory directly to the CPU. It also "pins" the memory, or page locks it, which will allow for asynchronous copying of the memory to and from a GPU. Too much pinned memory can interfere with CPU performance, so use it only with intention. Pinned memory should be freed with `cudaFreeHost`.
- `cudaMemcpy` can copy (not transfer) memory, either from host to device or from device to host.

以下是一些用于手动设备内存管理的CUDA命令：

- “cudaMalloc”将直接为活动的GPU分配内存。这可以防止所有GPU页面错误。在exchange中，它返回的指针不可由主机代码访问。

- “cudaMallocHost”将直接向CPU分配内存。它还“固定”内存，或页面锁定内存，这将允许异步复制内存到和从一个GPU。过多的固定内存会影响CPU的性能，所以只能有目的地使用它。应使用“cudaFreeHost”释放固定内存。

- “cudammcpy”可以从主机到设备或从设备到主机复制（而不是传输）内存。

### Manual Device Memory Management Example 手动设备内存管理示例

Here is a snippet of code that demonstrates the use of the above CUDA API calls.

下面是一段代码，演示了上述CUDAAPI调用的使用。

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

### Exercise: Manually Allocate Host and Device Memory 练习：手动分配主机和设备内存

The most recent iteration of the vector addition application, [01-stream-init-solution](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/06-stream-init/solutions/01-stream-init-solution.cu), is using `cudaMallocManaged` to allocate managed memory first used on the device by the initialization kernels, then on the device by the vector add kernel, and then by the host, where the memory is automatically transfered, for verification. This is a sensible approach, but it is worth experimenting with some manual device memory allocation and copying to observe its impact on the application's performance.

向量加法应用程序的最新迭代[01 stream init solution]使用“cudamalocmanaged”来分配初始化内核首先在设备上使用的托管内存，然后在设备上通过矢量添加内核，然后通过主机，在那里内存被自动传输，进行验证。这是一种明智的方法，但值得尝试一些手动设备内存分配和复制，以观察其对应用程序性能的影响。

Refactor the [01-stream-init-solution](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/06-stream-init/solutions/01-stream-init-solution.cu) application to **not** use `cudaMallocManaged`. In order to do this you will need to do the following:

- Replace calls to `cudaMallocManaged` with `cudaMalloc`.
- Create an additional vector that will be used for verification on the host. This is required since the memory allocated with `cudaMalloc` is not available to the host. Allocate this host vector with `cudaMallocHost`.
- After the `addVectorsInto` kernel completes, use `cudaMemcpy` to copy the vector with the addition results, into the host vector you created with `cudaMallocHost`.
- Use `cudaFreeHost` to free the memory allocated with `cudaMallocHost`.

重构[01 stream init solution]应用程序以**不**使用“cudamalocmanaged”。为此，您需要执行以下操作：

- 用“cudaMalloc”替换对“cudamalocmanaged”的调用。

- 创建将用于主机上验证的附加向量。这是必需的，因为用“cudaMalloc”分配的内存对主机不可用。使用“cudamalochost”分配此主机向量。

- 在“addVectorsInto”内核完成后，使用“cudammcpy”将包含添加结果的向量复制到使用“cudaMallocHost”创建的宿主向量中。

- 使用“cudaFreeHost”释放使用“cudamalochost”分配的内存。

Refer to [the solution](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu) if you get stuck.

如果遇到问题，请参阅[解决方案]

In [ ]:

```
!nvcc -o vector-add-manual-alloc 06-stream-init/solutions/01-stream-init-solution.cu -run
```

After completing the refactor, open a report in Nsight Systems, and use the timeline to do the following:

- Notice that there is no longer a *Unified Memory* section of the timeline.
- Comparing this timeline to that of the previous refactor, compare the runtimes of `cudaMalloc` in the current application vs. `cudaMallocManaged` in the previous.
- Notice how in the current application, work on the initialization kernels does not start until a later time than it did in the previous iteration. Examination of the timeline will show the difference is the time taken by `cudaMallocHost`. This clearly points out the difference between memory transfers, and memory copies. When copying memory, as you are doing presently, the data will exist in 2 different places in the system. In the current case, the allocation of the 4th host-only vector incurs a small cost in performance, compared to only allocating 3 vectors in the previous iteration.

完成重构后，在Nsight系统中打开一个报表，并使用时间线执行以下操作：

- 请注意，时间线中不再有“统一内存”部分。

- 将此时间线与上一个重构的时间线进行比较，将当前应用程序中的“cudaMalloc”与上一个应用程序中的“cudamalocmanaged”的运行时进行比较。

- 请注意，在当前应用程序中，初始化内核的工作要比前一次迭代中的工作晚一点才能开始。对时间线的检查将显示不同的是“cudamalochost”所花费的时间。这清楚地指出了内存传输和内存复制之间的区别。在复制内存时，正如您目前所做的，数据将存在于系统中的两个不同位置。在当前情况下，与在上一次迭代中仅分配3个向量相比，仅分配第4个主机向量在性能上花费较小。

------

## Using Streams to Overlap Data Transfers and Code Execution 使用流重叠数据传输和代码执行

The following slides present upcoming material visually, at a high level. Click through the slides before moving on to more detailed coverage of their topics in following sections.

以下幻灯片以高层次直观地展示了即将到来的材料。单击幻灯片，然后转到下面各节中对其主题的更详细介绍。

In [ ]:

```
%%HTML

<div align="center"><iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQdHDR62S4hhvq02CZreC_Hvb9y89_IRIKtCQQ-eMItim744eRHOK6Gead5P_EaPj66Z3_NS0hlTRuh/embed?start=false&loop=false&delayms=3000" frameborder="0" width="900" height="550" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe></div>
```

In addition to `cudaMemcpy` is `cudaMemcpyAsync` which can asynchronously copy memory either from host to device or from device to host as long as the host memory is pinned, which can be done by allocating it with `cudaMallocHost`.

除了“cudammcpy”之外，还有“cudammcpyasync”，它可以在主机到设备或设备到主机之间异步复制内存，只要主机内存被固定，就可以使用“cudaMallocHost”分配内存。

Similar to kernel execution, `cudaMemcpyAsync` is only asynchronous by default with respect to the host. It executes, by default, in the default stream and therefore is a blocking operation with regard to other CUDA operations occuring on the GPU. The `cudaMemcpyAsync` function, however, takes as an optional 5th argument, a non-default stream. By passing it a non-default stream, the memory transfer can be concurrent to other CUDA operations occuring in other non-default streams.

与内核执行类似，“cudaMemcpyAsync”默认情况下仅与主机异步。默认情况下，它在默认流中执行，因此是与GPU上发生的其他CUDA操作相关的阻塞操作。但是，“cudammcpyasync”函数将第5个参数（非默认流）作为可选参数。通过向它传递一个非默认流，内存传输可以并发到其他非默认流中发生的其他CUDA操作。

A common and useful pattern is to use a combination of pinned host memory, asynchronous memory copies in non-default streams, and kernel executions in non-default streams, to overlap memory transfers with kernel execution.

一种常见且有用的模式是使用固定主机内存、非默认流中的异步内存副本和非默认流中的内核执行的组合，以使内存传输与内核执行重叠。

In the following example, rather than wait for the entire memory copy to complete before beginning work on the kernel, segments of the required data are copied and worked on, with each copy/work segment running in its own non-default stream. Using this technique, work on parts of the data can begin while memory transfers for later segments occur concurrently. Extra care must be taken when using this technique to calculate segment-specific values for the number of operations, and the offset location inside arrays, as shown here:

在下面的示例中，不是等待整个内存复制完成后才开始对内核进行操作，而是复制和处理所需数据的段，每个复制/工作段在其自己的非默认流中运行。使用这种技术，可以在并发地为以后的段进行内存传输时开始处理部分数据。使用此技术计算特定于段的操作数值和数组内的偏移位置时，必须格外小心，如下所示：

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

### Exercise: Overlap Kernel Execution and Memory Copy Back to Host 练习：重叠内核执行和将内存复制回主机

The most recent iteration of the vector addition application, [01-manual-malloc-solution.cu](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu), is currently performing all of its vector addition work on the GPU before copying the memory back to the host for verification.

向量添加应用程序的最新迭代[01 manual malloc solution.cu]目前正在GPU上执行其所有向量添加工作，然后将内存复制回主机进行验证。

Refactor [01-manual-malloc-solution.cu](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/07-manual-malloc/solutions/01-manual-malloc-solution.cu) to perform the vector addition in 4 segments, in non-default streams, so that asynchronous memory copies can begin before waiting for all vector addition work to complete. Refer to [the solution](http://ec2-3-85-242-54.compute-1.amazonaws.com/lab/edit/08-overlap-xfer/solutions/01-overlap-xfer-solution.cu) if you get stuck.

重构[01 manual malloc solution.cu]以在4个段中执行向量添加，在非默认流中，以便在等待所有向量添加工作完成之前可以开始异步内存复制。如果遇到问题，请参阅[解决方案]

In [ ]:

```
!nvcc -o vector-add-manual-alloc 07-manual-malloc/solutions/01-manual-malloc-solution.cu -run
```

After completing the refactor, open a report in Nsight Systems, and use the timeline to do the following:

- Note when the device to host memory transfers begin, is it before or after all kernel work has completed?
- Notice that the 4 memory copy segments themselves do not overlap. Even in separate non-default streams, only one memory transfer in a given direction (DtoH here) at a time can occur simultaneously. The performance gains here are in the ability to start the transfers earlier than otherwise, and it is not hard to imagine in an application where a less trivial amount of work was being done compared to a simple addition operation, that the memory copies would not only start earlier, but also overlap with kernel execution.

完成重构后，在Nsight系统中打开一个报表，并使用时间线执行以下操作：

- 注意：当设备到主机的内存传输开始时，是在所有内核工作完成之前还是之后？

- 请注意，4个内存复制段本身并不重叠。即使在单独的非默认流中，一次只能在给定方向（这里是DtoH）上同时发生一次内存传输。这里的性能提高在于能够比其他方式更早地开始传输，而且在一个与简单的加法操作相比所做的工作更少的应用程序中，不难想象内存拷贝不仅会更早地开始，而且还会与内核执行重叠。
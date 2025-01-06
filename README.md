# Parallel and Distributed Hough Transform Implementation

This project implements the Hough Transform for image processing in a parallel and distributed manner, utilizing three approaches:
- **Threads** (Multithreading).
- **MPI** (Message Passing Interface).
- **CUDA** (GPU acceleration).

By leveraging these methods, we significantly optimized the computation time required for performing the Hough Transform.

## Overview
The Hough Transform is a technique used to detect shapes in images, such as lines and circles. This implementation focuses on distributing the computational workload to improve performance.

### Key Features
- **Multithreading:** Utilized normal threads to distribute the workload across CPU cores.
- **Distributed Computing with MPI:** Split the image into chunks and processed them across multiple machines or processes.
- **GPU Acceleration with CUDA:** Leveraged the parallel processing power of GPUs to enhance computation speed.

## Technical Details
- **Programming Language:** C++
- **Libraries and Frameworks:**
  - OpenCV: For loading images and dividing them into manageable chunks.
  - MPI: For distributed computation.
  - CUDA: For GPU-based parallel processing.

## Methodology
1. **Image Preprocessing:**
   - Loaded images using OpenCV.
   - Split the image into smaller chunks for parallel processing.

2. **Workload Distribution:**
   - For **Threads**: Divided work among CPU threads.
   - For **MPI**: Distributed chunks to different processes or machines.
   - For **CUDA**: Offloaded computation to the GPU for massively parallel execution.

3. **Hough Transform Computation:**
   - Performed the core computation using optimized algorithms tailored to the specific approach (Threads, MPI, or CUDA).

4. **Result Aggregation:**
   - Combined results from different workers to reconstruct the final output image.

## Conclusion
This project demonstrates how the Hough Transform can be efficiently implemented using parallel and distributed computing techniques. The use of multithreading, distributed processing, and GPU acceleration showcases the versatility of modern computational tools in optimizing traditional algorithms.


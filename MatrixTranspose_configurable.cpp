/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <cstdlib>
#include <cmath>

// hip header file
#include "hip/hip_runtime.h"

#define WIDTH     1024
#define NUM       (WIDTH*WIDTH)

#define THREADS_PER_BLOCK_X  4
#define THREADS_PER_BLOCK_Y  4
#define THREADS_PER_BLOCK_Z  1

// Device (Kernel) function, it must be void
// hipLaunchParm provides the execution configuration
__global__ void matrixTranspose(float *out,
                                float *in,
                                const int width)
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void matrixTransposeCPUReference(
    float * output,
    float * input,
    const unsigned int width)
{
    for(unsigned int j = 0; j < width; j++)
    {
        for(unsigned int i = 0; i < width; i++)
        {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int main(int argc, char** argv) {

  float* Matrix;
  float* TransposeMatrix;
  float* cpuTransposeMatrix;

  float* gpuMatrix;
  float* gpuTransposeMatrix;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  std::cout << "Device name " << devProp.name << std::endl;

  // Runtime-configurable parameters with defaults from macros
  int width  = WIDTH;
  int blockX = THREADS_PER_BLOCK_X;
  int blockY = THREADS_PER_BLOCK_Y;

  if (argc > 1) width  = std::atoi(argv[1]);
  if (argc > 2) blockX = std::atoi(argv[2]);
  if (argc > 3) blockY = std::atoi(argv[3]);

  if (width <= 0 || blockX <= 0 || blockY <= 0 ||
      (width % blockX) != 0 || (width % blockY) != 0) {
    std::cerr << "Usage: " << argv[0]
              << " [width block_x block_y]\n"
              << "Constraints: width > 0, block_x > 0, block_y > 0, "
              << "width % block_x == 0, width % block_y == 0\n";
    return 1;
  }

  size_t numElements = static_cast<size_t>(width) * static_cast<size_t>(width);

  std::cout << "Using width=" << width
            << " blockX=" << blockX
            << " blockY=" << blockY << std::endl;

  int errors = 0;

  // allocate host memory
  hipHostMalloc(&Matrix, numElements * sizeof(float));
  TransposeMatrix    = (float*)malloc(numElements * sizeof(float));
  cpuTransposeMatrix = (float*)malloc(numElements * sizeof(float));

  // initialize the input data
  for (size_t idx = 0; idx < numElements; ++idx) {
    Matrix[idx] = (float)idx * 10.0f;
  }

  // allocate the memory on the device side
  hipMalloc((void**)&gpuMatrix,          numElements * sizeof(float));
  hipMalloc((void**)&gpuTransposeMatrix, numElements * sizeof(float));

  // Memory transfer from host to device
  hipMemcpy(gpuMatrix, Matrix,
            numElements * sizeof(float),
            hipMemcpyHostToDevice);

  // Lauching kernel from host
  dim3 grid(width / blockX, width / blockY);
  dim3 block(blockX, blockY);

  hipLaunchKernelGGL(matrixTranspose,
                     grid,
                     block,
                     0, 0,
                     gpuTransposeMatrix, gpuMatrix, width);

  // Memory transfer from device to host
  hipMemcpy(TransposeMatrix, gpuTransposeMatrix,
            numElements * sizeof(float),
            hipMemcpyDeviceToHost);

  // CPU MatrixTranspose computation
  matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, width);

  // verify the results
  double eps = 1.0E-6;
  for (size_t idx = 0; idx < numElements; ++idx) {
    if (std::abs(TransposeMatrix[idx] - cpuTransposeMatrix[idx]) > eps) {
      errors++;
    }
  }

  if (errors != 0) {
    printf("FAILED: %d errors\n", errors);
  } else {
    printf("PASSED!\n");
  }

  // free the resources on device side
  hipFree(gpuMatrix);
  hipFree(gpuTransposeMatrix);

  // free the resources on host side
  hipFree(Matrix);
  free(TransposeMatrix);
  free(cpuTransposeMatrix);

  return errors;
}

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

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Forward declarations for functions in other files
__device__ void processDataBlock(float *input, float *output, int idx);
__device__ void finalTransform(float *input, float *output, int idx);

// Kernel that uses shared memory and calls functions from other files
__global__ void sharedMemoryKernel(float *input, float *output, int size) {
  // Kernel's static shared memory
  __shared__ float kernelShared[256];

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Load data into kernel's shared memory
  if (idx < size) {
    kernelShared[threadIdx.x] = input[idx];
  }

  // Ensure all threads have loaded their data
  __syncthreads();

  if (idx < size) {
    float tempResult;

    // First, call processDataBlock from src1.cpp - it has its own shared memory
    processDataBlock(kernelShared, &tempResult, threadIdx.x);

    // Ensure all processing is complete
    __syncthreads();

    // Update kernel's shared memory with intermediate result
    kernelShared[threadIdx.x] = tempResult;

    __syncthreads();

    // Then, call finalTransform from src2.cpp - it also has its own shared
    // memory
    finalTransform(kernelShared, &tempResult, threadIdx.x);

    // Write the final result to global memory
    output[idx] = tempResult;
  }
}

int main() {
  const int size = 1024;
  const int bytes = size * sizeof(float);

  // Host data
  std::vector<float> h_input(size);
  std::vector<float> h_output(size);

  // Initialize input data
  for (int i = 0; i < size; i++) {
    h_input[i] = static_cast<float>(i);
  }

  // Device data
  float *d_input, *d_output;
  hipMalloc(&d_input, bytes);
  hipMalloc(&d_output, bytes);

  // Copy data to device
  hipMemcpy(d_input, h_input.data(), bytes, hipMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  hipLaunchKernelGGL(sharedMemoryKernel, dim3(blocksPerGrid),
                     dim3(threadsPerBlock), 0, 0, d_input, d_output, size);

  // Check for errors
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::cerr << "Kernel launch failed: " << hipGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Copy results back to host
  hipMemcpy(h_output.data(), d_output, bytes, hipMemcpyDeviceToHost);

  // Verify results
  bool passed = true;
  std::cout << "Verifying results..." << std::endl;

  for (int i = 0; i < size; i++) {
    // Expected result from operations across all functions
    float expected = h_input[i] * 2.0f + 10.0f;

    if (fabs(h_output[i] - expected) > 1e-5) {
      std::cout << "Error at position " << i << ": " << h_output[i]
                << " != " << expected << std::endl;
      passed = false;
      if (i > 10)
        break; // Limit error output
    }
  }

  if (passed) {
    std::cout << "All tests passed! Static shared memory variables work "
                 "correctly across files."
              << std::endl;
  } else {
    std::cout << "Tests failed!" << std::endl;
  }

  // Cleanup
  hipFree(d_input);
  hipFree(d_output);

  return 0;
}

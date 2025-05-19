#include <hip/hip_runtime.h>
#include <iostream>

// Forward declarations of functions from other files
__device__ void deviceFunction1(float *data, int idx);
__device__ void initializeFunctionPointer(int value);

// Main kernel that orchestrates the function calls
__global__ void processingKernel(float *data, int size, int operationType) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < size) {
    // First, initialize the function pointer in src2.cpp
    // This will set the global function pointer to point to either
    // multiplyByTwo or multiplyByThree depending on operationType
    initializeFunctionPointer(operationType);

    // Make sure all threads finish initialization before proceeding
    __syncthreads();

    // Now call deviceFunction1 from src1.cpp, which will use the function
    // pointer
    deviceFunction1(data, idx);
  }
}

int main() {
  const int size = 256;
  const int bytes = size * sizeof(float);

  // Host data
  float *h_data = new float[size];
  for (int i = 0; i < size; i++) {
    h_data[i] = static_cast<float>(i);
  }

  // Device data
  float *d_data;
  hipMalloc(&d_data, bytes);
  hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);

  // Launch kernel with multiply-by-two operation (operationType = 2)
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  int operationType = 2; // Use multiplication by 2

  hipLaunchKernelGGL(processingKernel, dim3(blocksPerGrid),
                     dim3(threadsPerBlock), 0, 0, d_data, size, operationType);

  // Check for errors
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::cerr << "Kernel launch failed: " << hipGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Copy results back
  hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost);

  // Verify results - should be original values multiplied by 2
  // Verify results - should be original values multiplied by 2 plus the
  // modifier (2 * 0.1 = 0.2)
  bool passedFirstTest = true;
  for (int i = 0; i < size; i++) {
    float expected = i * 2.0f + 0.2f;
    if (fabs(h_data[i] - expected) > 1e-5) {
      std::cout << "Error at position " << i << ": " << h_data[i]
                << " != " << expected << std::endl;
      passedFirstTest = false;
      break;
    }
  }

  if (passedFirstTest) {
    std::cout
        << "First test passed! Multiply-by-two operation worked correctly."
        << std::endl;
  }

  // Reset the data
  for (int i = 0; i < size; i++) {
    h_data[i] = static_cast<float>(i);
  }
  hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);

  // Now try with multiply-by-three operation (operationType = 3)
  operationType = 3;

  hipLaunchKernelGGL(processingKernel, dim3(blocksPerGrid),
                     dim3(threadsPerBlock), 0, 0, d_data, size, operationType);

  // Check for errors
  err = hipGetLastError();
  if (err != hipSuccess) {
    std::cerr << "Kernel launch failed: " << hipGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Copy results back
  hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost);

  // Verify results - should be original values multiplied by 3 plus the
  // modifier (3 * 0.1 = 0.3)
  bool passedSecondTest = true;
  for (int i = 0; i < size; i++) {
    float expected = i * 3.0f + 0.3f;
    if (fabs(h_data[i] - expected) > 1e-5) {
      std::cout << "Error at position " << i << ": " << h_data[i]
                << " != " << expected << std::endl;
      passedSecondTest = false;
      break;
    }
  }

  if (passedSecondTest) {
    std::cout
        << "Second test passed! Multiply-by-three operation worked correctly."
        << std::endl;
  }

  if (passedFirstTest && passedSecondTest) {
    std::cout << "All tests passed! The function pointer mechanism works "
                 "across all three files."
              << std::endl;
  }

  // Cleanup
  delete[] h_data;
  hipFree(d_data);

  return 0;
}

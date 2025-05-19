#include <hip/hip_runtime.h>
#include <iostream>

// Forward declaration of the device function in other1.hip
__device__ void foo(int s);

// Forward declarations of result-checking functions
__device__ int getDerived1Result();
__device__ int getDerived2Result();

// Kernel that calls the device function
__global__ void kernel(int s) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    foo(s);
  }
}

// Kernels to check results
__global__ void checkDerived1Result(int *result) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *result = getDerived1Result();
  }
}

__global__ void checkDerived2Result(int *result) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *result = getDerived2Result();
  }
}

int main() {
  // Allocate memory for results
  int *d_result;
  hipMalloc(&d_result, sizeof(int));

  int h_result = 0;

  // First test: use Derived1 (s = 1)
  hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, 0, 1);

  // Check for errors
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    std::cerr << "Kernel launch failed: " << hipGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Wait for kernel to finish
  hipDeviceSynchronize();

  // Check Derived1 result
  hipLaunchKernelGGL(checkDerived1Result, dim3(1), dim3(1), 0, 0, d_result);

  hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);

  std::cout << "Test with Derived1 (s=1): ";
  if (h_result == 100) {
    std::cout << "PASSED! Derived1::result = " << h_result << std::endl;
  } else {
    std::cout << "FAILED! Expected 100, got " << h_result << std::endl;
  }

  // Second test: use Derived2 (s = 0)
  hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, 0, 0);

  // Check for errors
  err = hipGetLastError();
  if (err != hipSuccess) {
    std::cerr << "Kernel launch failed: " << hipGetErrorString(err)
              << std::endl;
    return -1;
  }

  // Wait for kernel to finish
  hipDeviceSynchronize();

  // Check Derived2 result
  hipLaunchKernelGGL(checkDerived2Result, dim3(1), dim3(1), 0, 0, d_result);

  hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);

  std::cout << "Test with Derived2 (s=0): ";
  if (h_result == 200) {
    std::cout << "PASSED! Derived2::result = " << h_result << std::endl;
  } else {
    std::cout << "FAILED! Expected 200, got " << h_result << std::endl;
  }

  // Cleanup
  hipFree(d_result);

  return 0;
}

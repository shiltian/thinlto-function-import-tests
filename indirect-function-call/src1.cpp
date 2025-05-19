#include <hip/hip_runtime.h>

// Define a function pointer type for our operation functions
typedef void (*MathOperationFunction)(float *, int);

// Global function pointer that will be set by src2.cpp
__device__ MathOperationFunction g_mathOperationPtr = nullptr;

// Device function that uses the function pointer
__device__ void deviceFunction1(float *data, int idx) {
  if (g_mathOperationPtr != nullptr) {
    // Call whatever function g_mathOperationPtr points to
    g_mathOperationPtr(data, idx);
  }
}

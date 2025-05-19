#include <hip/hip_runtime.h>

// Define the function pointer type (same as in src1.hip)
typedef void (*MathOperationFunction)(float *, int);

// External reference to the global function pointer defined in src1.hip
extern __device__ MathOperationFunction g_mathOperationPtr;

// Static (file-scope) global variable to store an operation modifier
static __device__ float s_operationModifier = 0.0f;

// First operation function - multiply by two plus the modifier
__device__ void multiplyByTwo(float *data, int idx) {
  // Use the static global modifier to adjust the result
  data[idx] = data[idx] * 2.0f + s_operationModifier;
}

// Second operation function - multiply by three plus the modifier
__device__ void multiplyByThree(float *data, int idx) {
  // Use the static global modifier to adjust the result
  data[idx] = data[idx] * 3.0f + s_operationModifier;
}

// Function to initialize the function pointer based on the operation type
// Also sets the static global operation modifier
__device__ void initializeFunctionPointer(int operationType) {
  // Set the operation modifier based on the operation type
  // For demonstration: add a small offset to each result
  s_operationModifier = operationType * 0.1f;

  if (operationType == 2) {
    g_mathOperationPtr = multiplyByTwo;
  } else if (operationType == 3) {
    g_mathOperationPtr = multiplyByThree;
  } else {
    // Default to multiply by two
    g_mathOperationPtr = multiplyByTwo;
  }
}

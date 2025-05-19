#include <hip/hip_runtime.h>

// Forward declaration for function in the same file
__device__ void multiplyData(float *tempData, float *result, int idx);

// Function that processes data using its own static shared memory
__device__ void processDataBlock(float *inputData, float *result, int idx) {
  // Static shared memory for this function
  __shared__ float processShared[256];

  // Copy data from input to our own shared memory
  processShared[idx] = inputData[idx];

  // Ensure all threads have copied their data
  __syncthreads();

  // Do some initial processing
  processShared[idx] += 1.0f;

  // Ensure all threads complete the initial processing
  __syncthreads();

  // Call another function within the same file
  multiplyData(processShared, result, idx);
}

// Helper function that further processes the data with its own static shared
// memory
__device__ void multiplyData(float *tempData, float *result, int idx) {
  // Static shared memory for this function
  __shared__ float multiplyShared[256];

  // Copy data from the previous function's shared memory
  multiplyShared[idx] = tempData[idx];

  // Ensure all threads have copied their data
  __syncthreads();

  // Multiply the data by 2
  multiplyShared[idx] = multiplyShared[idx] * 2.0f;

  // Ensure all threads complete their multiplication
  __syncthreads();

  // Write result to the output parameter
  *result = multiplyShared[idx];
}

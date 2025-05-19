#include <hip/hip_runtime.h>

// Function that applies a final transformation with its own static shared
// memory
__device__ void finalTransform(float *inputData, float *result, int idx) {
  // Static shared memory for this function
  __shared__ float transformShared[256];

  // Copy data from input to our own shared memory
  transformShared[idx] = inputData[idx];

  // Ensure all threads have copied their data
  __syncthreads();

  // Add 10 to each element
  transformShared[idx] += 10.0f;

  // Optional: Do some neighborhood operations to demonstrate shared memory
  if (idx > 0 && idx < 255) {
    // Just accessing neighboring elements to show shared memory usage
    // This operation doesn't affect the final result for verification
    float temp = transformShared[idx - 1] + transformShared[idx + 1];
    // Just to ensure the compiler doesn't optimize away the calculation
    if (temp > 1000000.0f) {
      transformShared[idx] += 0.000001f; // Negligible for verification
    }
  }

  // Ensure all transformations are complete
  __syncthreads();

  // Write result to the output parameter
  *result = transformShared[idx];
}

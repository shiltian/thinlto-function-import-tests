#include "class-def.h"
#include <hip/hip_runtime.h>

// Forward declaration of the device function in other2.hip
__device__ void bar(Base *ptr);

// Helper functions to access results from main.hip
__device__ int getDerived1Result() { return Derived1::result; }

__device__ int getDerived2Result() { return Derived2::result; }

// Implementation of device function foo that creates objects
__device__ void foo(int s) {
  // Create either Derived1 or Derived2 based on parameter s
  Base *ptr = nullptr;
  if (s)
    ptr = new Derived1();
  else
    ptr = new Derived2();

  // Call the device function in other2.hip
  bar(ptr);

  // Clean up
  delete ptr;
}

// Initialize static member
__device__ int Derived1::result = 0;
__device__ int Derived2::result = 0;

#include "class-def.h"
#include <hip/hip_runtime.h>

// Implementation of device function bar that calls the virtual function
__device__ void bar(Base *ptr) {
  // Call the virtual function - will execute either Derived1::vfunc or
  // Derived2::vfunc
  ptr->vfunc();
}

#ifndef CLASS_DEF_H
#define CLASS_DEF_H

// Base class with virtual function
class __device__ Base {
public:
  __device__ virtual void vfunc() = 0;
  __device__ virtual ~Base() {}
};

// First derived class implementation
class __device__ Derived1 : public Base {
public:
  __device__ Derived1() {}

  __device__ void vfunc() override {
    // Implementation for Derived1
    result = 100;
  }

  static __device__ int result;
};

// Second derived class implementation
class __device__ Derived2 : public Base {
public:
  __device__ Derived2() {}

  __device__ void vfunc() override {
    // Implementation for Derived2
    result = 200;
  }

  static __device__ int result;
};

#endif // CLASS_DEF_H

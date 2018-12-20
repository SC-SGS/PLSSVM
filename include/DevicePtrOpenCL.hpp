#pragma once

#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include <CL/cl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace opencl {

template <class T> class DevicePtrOpenCL {
public:
  DevicePtrOpenCL() : ptr(nullptr), buffer_size(0){};

  DevicePtrOpenCL(device_t &device, const int buffer_size)
      : device(device), buffer_size(buffer_size) {
    cl_int err;
    ptr = clCreateBuffer(device.context, CL_MEM_READ_WRITE,
                         sizeof(T) * buffer_size, nullptr, &err);
    opencl::check(err, "DevicePtrOpenCL: clCreateBuffer failed");
  }
  DevicePtrOpenCL(const DevicePtrOpenCL &other) = delete;
  ~DevicePtrOpenCL() {
    if (ptr) {
      clReleaseMemObject(ptr);
      ptr = nullptr;
    }
  }

  DevicePtrOpenCL &operator=(const DevicePtrOpenCL &) = delete;

  DevicePtrOpenCL(DevicePtrOpenCL &&other) {
    this->device = other.device;
    this->ptr = other.ptr;
    other.ptr = nullptr;
    this->buffer_size = other.buffer_size;
    other.buffer_size = 0;
  }

  DevicePtrOpenCL &operator=(DevicePtrOpenCL &&other) {
    this->device = other.device;
    this->ptr = other.ptr;
    other.ptr = nullptr;
    this->buffer_size = other.buffer_size;
    other.buffer_size = 0;
    return *this;
  }

  cl_mem get() { return ptr; }

  void to_device(const std::vector<T> data) {
    if (!ptr) {
      throw std::runtime_error("DevicePtrOpenCL: buffer not initialized");
    }
    if (data.size() != buffer_size) {
      throw std::runtime_error(
          "DevicePtrOpenCL: to_device: host buffer and device "
          "buffer sizes don't match");
    }
    cl_int err;
    err = clEnqueueWriteBuffer(device.commandQueue, ptr, CL_TRUE, 0,
                               sizeof(T) * buffer_size, data.data(), 0, nullptr,
                               nullptr);
    opencl::check(err, "DevicePtrOpenCL: clEnqueueWriteBuffer failed");
    clFinish(device.commandQueue);
  }

  void from_device(std::vector<T> &data) {
    if (!ptr) {
      throw std::runtime_error("DevicePtrOpenCL: buffer not initialized");
    }
    if (data.size() != buffer_size) {
      throw std::runtime_error(
          "DevicePtrOpenCL: from_device: host buffer and device "
          "buffer sizes don't match");
    }
    cl_int err;
    err = clEnqueueReadBuffer(device.commandQueue, ptr, CL_TRUE, 0,
                              sizeof(T) * buffer_size, data.data(), 0, nullptr,
                              nullptr);
    opencl::check(err, "DevicePtrOpenCL: clEnqueueReadBuffer failed");
    clFinish(device.commandQueue);
  }

  void fill_buffer(T value) {
    if (!ptr) {
      throw std::runtime_error("DevicePtrOpenCL: buffer not initialized");
    }
    cl_int err;
    err = clEnqueueFillBuffer(device.commandQueue, ptr, &value, sizeof(value),
                              0, buffer_size * sizeof(T), 0, nullptr, nullptr);
    opencl::check(err, "DevicePtrOpenCL: clEnqueueFillBuffer failed");
    clFinish(device.commandQueue);
  }


  void resize(const size_t size, T value = static_cast<T>(0.0)) {
    if (!ptr) {
      throw std::runtime_error("DevicePtrOpenCL: buffer not initialized");
    }
    if(size != buffer_size){
      std::vector<T> buffer(buffer_size);
      from_device(buffer);
      buffer_size = size;
      clReleaseMemObject(ptr);
      cl_int err;
      ptr = clCreateBuffer(device.context, CL_MEM_READ_WRITE,
                          sizeof(T) * buffer_size, nullptr, &err);
      opencl::check(err, "DevicePtrOpenCL: resize failed");
      buffer.resize(buffer_size, value);
      to_device(buffer);
    }
  }

private:
  device_t device;
  cl_mem ptr;
  size_t buffer_size;
};
} // namespace opencl

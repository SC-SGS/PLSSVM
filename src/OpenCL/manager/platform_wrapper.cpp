#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "manager_error.hpp"
#include "platform_wrapper.hpp"

namespace opencl {

platform_wrapper_t::platform_wrapper_t(
    cl_platform_id platformId, char (&platformName)[128],
    const std::vector<cl_device_id> &deviceIds,
    const std::vector<std::string> &deviceNames)
    : platformId(platformId), deviceIds(deviceIds), deviceNames(deviceNames) {
  for (size_t i = 0; i < 128; i++) {
    this->platformName[i] = platformName[i];
  }

  cl_int err = CL_SUCCESS;
  // Create OpenCL context
  cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platformId, 0};

  this->context =
      clCreateContext(properties, (cl_uint)this->deviceIds.size(),
                      this->deviceIds.data(), nullptr, nullptr, &err);

  if (err != CL_SUCCESS) {
    std::stringstream errorString;
    errorString << "OCL Error: Failed to create OpenCL context! Error Code: "
                << err << std::endl;
    throw manager_error(errorString.str());
  }

  // Create a command queue for each device
  for (uint32_t i = 0; i < this->deviceIds.size(); i++) {
    this->commandQueues.push_back(clCreateCommandQueue(
        this->context, this->deviceIds[i], CL_QUEUE_PROFILING_ENABLE, &err));

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create command queue! Error Code: "
                  << err << std::endl;
      throw manager_error(errorString.str());
    }

    char deviceName[128] = {0};
    err = clGetDeviceInfo(this->deviceIds[i], CL_DEVICE_NAME,
                          128 * sizeof(char), &deviceName, nullptr);

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to read the device name for device: "
                  << this->deviceIds[i] << std::endl;
      throw manager_error(errorString.str());
    }
    this->deviceNames.push_back(deviceName);
  }
}

platform_wrapper_t::platform_wrapper_t(const platform_wrapper_t &original)
    : platformId(original.platformId), context(original.context),
      deviceIds(original.deviceIds), deviceNames(original.deviceNames),
      commandQueues(original.commandQueues) {
  for (size_t i = 0; i < 128; i++) {
    platformName[i] = original.platformName[i];
  }

  // increment the reference counter of contexts and command queues

  cl_int err = CL_SUCCESS;
  // Create a command queue for each device
  for (uint32_t i = 0; i < this->deviceIds.size(); i++) {
    err = clRetainCommandQueue(this->commandQueues[i]);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Could not release command queue! Error Code: "
                  << err << std::endl;
      throw manager_error(errorString.str());
    }
  }

  err = clRetainContext(this->context);
  if (err != CL_SUCCESS) {
    std::stringstream errorString;
    errorString << "OCL Error: Could not release context! Error Code: " << err
                << std::endl;
    throw manager_error(errorString.str());
  }
}

platform_wrapper_t::~platform_wrapper_t() {
  cl_int err = CL_SUCCESS;
  // Create a command queue for each device
  for (uint32_t i = 0; i < this->deviceIds.size(); i++) {
    err = clReleaseCommandQueue(this->commandQueues[i]);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      std::cerr << "OCL Error: Could not release command queue! Error Code: "
                << err << std::endl;
      std::terminate();
    }
  }

  err = clReleaseContext(this->context);
  if (err != CL_SUCCESS) {
    std::stringstream errorString;
    std::cerr << "OCL Error: Could not release context! Error Code: " << err
              << std::endl;
    std::terminate();
  }
}

size_t platform_wrapper_t::getDeviceCount() { return this->deviceIds.size(); }
} // namespace opencl

#pragma once

#include <string>
#include <vector>

#include "CL/cl.h"

namespace opencl {

class platform_wrapper_t {
public:
  cl_platform_id platformId;
  char platformName[128];
  cl_context context;
  std::vector<cl_device_id> deviceIds;
  std::vector<std::string> deviceNames;
  std::vector<cl_command_queue> commandQueues;

  platform_wrapper_t(cl_platform_id platformId, char (&platformName)[128],
                     const std::vector<cl_device_id> &deviceIds,
                     const std::vector<std::string> &deviceName);

  platform_wrapper_t(const platform_wrapper_t &original);

  platform_wrapper_t &operator=(const platform_wrapper_t &other) = delete;

  ~platform_wrapper_t();

  size_t getDeviceCount();
};
} // namespace opencl

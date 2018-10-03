#pragma once

// define required for clCreateCommandQueue on platforms that don't support
// OCL2.0 yet
// #define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.h>

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "device.hpp"
#include "error_codes.hpp"
#include "manager_error.hpp"
#include "platform_wrapper.hpp"

namespace opencl {

class manager_t {
public:
  configuration_t parameters;

  std::vector<platform_wrapper_t> platforms;

  // linear device list
  std::vector<device_t> devices;

  cl_uint overallDeviceCount; // devices over all platforms

  bool verbose;

public:
  explicit manager_t();

  explicit manager_t(const std::string &configuration_file_name);

  explicit manager_t(const configuration_t &parameters);

  ~manager_t();

  // for all devices
  void build_kernel(const std::string &program_src,
                    const std::string &kernel_name,
                    std::map<cl_platform_id, std::vector<cl_kernel>> &kernels);

  // for a single device
  cl_kernel build_kernel(const std::string &source, device_t &device,
                         json::node &kernelConfiguration,
                         const std::string &kernel_name);

  void configure(bool useConfiguration = false);

  void configure_platform(cl_platform_id platformId,
                          configuration_t &configuration,
                          bool useConfiguration);

  void configure_device(cl_device_id deviceId, json::node &devicesNode,
                        std::vector<cl_device_id> &filteredDeviceIds,
                        std::vector<std::string> &filteredDeviceNames,
                        std::map<std::string, size_t> &countLimitMap,
                        bool useConfiguration);

  configuration_t &get_configuration();

  std::vector<device_t> &get_devices();

  void set_verbose(bool verbose);

  std::string read_src_file(const std::string &kernel_src_file_name) const;
};

inline void check(cl_int err, const std::string &message) {
  if (err != CL_SUCCESS) {
    std::stringstream errorString;
    errorString << message << ", OpenCL error: " << resolve_error(err)
                << std::endl;
    throw manager_error(errorString.str());
  }
}

} // namespace opencl

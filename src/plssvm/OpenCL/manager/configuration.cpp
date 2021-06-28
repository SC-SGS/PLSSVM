#include "configuration.hpp"

#include <string>
#include <vector>

namespace opencl {

configuration_t::configuration_t() : json::json() {}

configuration_t::configuration_t(const std::string &file_name)
    : json::json(file_name) {}

configuration_t *configuration_t::clone() {
  return dynamic_cast<configuration_t *>(new configuration_t(*this));
}

std::vector<std::reference_wrapper<json::node>>
configuration_t::get_all_device_nodes() {
  std::vector<std::reference_wrapper<json::node>> deviceNodes;
  for (std::string &platformName : (*this)["PLATFORMS"].keys()) {
    json::node &platformNode = (*this)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::node &deviceNode = platformNode["DEVICES"][deviceName];
      deviceNodes.push_back(deviceNode);
    }
  }
  return deviceNodes;
}

std::unique_ptr<configuration_t>
configuration_t::from_string(std::string &parameters_string) {
  std::unique_ptr<configuration_t> parameters =
      std::make_unique<configuration_t>();
  parameters->deserializeFromString(parameters_string);
  return parameters;
}

} // namespace opencl

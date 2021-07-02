#pragma once

#include <map>
#include <string>
#include <vector>

#include "json/json.hpp"

namespace opencl {

class configuration_t : public json::json {
  public:
    configuration_t();

    explicit configuration_t(const std::string &file_name);

    configuration_t *clone() override;

    std::vector<std::reference_wrapper<json::node>> get_all_device_nodes();

    static std::unique_ptr<configuration_t>
    from_string(std::string &parameters_string);
};

}  // namespace opencl

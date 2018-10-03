#pragma once

#include <stdexcept>

namespace opencl {

class manager_error : public std::runtime_error {
public:
  manager_error(const std::string &what) : std::runtime_error(what) {}
};
} // namespace opencl

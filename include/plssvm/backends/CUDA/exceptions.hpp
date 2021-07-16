#pragma once

#include "plssvm/exceptions/exceptions.hpp"
#include "plssvm/exceptions/source_location.hpp"

#include <string>

namespace plssvm {

class cuda_backend_exception : public exception {
  public:
    explicit cuda_backend_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg + std::string(loc.file_name()) + std::string(loc.function_name()) + std::to_string(loc.line()), "cuda_backend_exception", loc } {}
};

};  // namespace plssvm
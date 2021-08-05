/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements custom exception classes specific to the OpenCL backend.
 */

#pragma once

#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include <string>  // std::string

namespace plssvm::opencl {

/**
 * @brief Exception type thrown if a problem with the OpenCL backend occurs.
 */
class backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit backend_exception(const std::string &msg, source_location loc = source_location::current()) :
        ::plssvm::exception{ msg, "opencl::backend_exception", loc } {}
};

};  // namespace plssvm::opencl
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements custom exception classes specific to the CUDA backend.
 */

#pragma once

#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include <string>  // std::string

namespace plssvm {

/**
 * @brief Exception type thrown if a problem with the CUDA backend occurs.
 */
class cuda_backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit cuda_backend_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "cuda_backend_exception", loc } {}
};

};  // namespace plssvm
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements custom exception classes specific to the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_EXCEPTIONS_HPP_
#define PLSSVM_BACKENDS_OPENMP_EXCEPTIONS_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include <string>  // std::string

namespace plssvm::openmp {

/**
 * @brief Exception type thrown if a problem with the OpenMP backend occurs.
 */
class backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit backend_exception(const std::string &msg, source_location loc = source_location::current());
};

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_EXCEPTIONS_HPP_
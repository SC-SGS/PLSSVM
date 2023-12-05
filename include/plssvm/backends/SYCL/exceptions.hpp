/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements custom exception classes specific to the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CSVM_HPP_
#define PLSSVM_BACKENDS_SYCL_CSVM_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include <string>  // std::string

namespace plssvm {

namespace sycl {

/**
 * @brief Exception type thrown if a problem with the SYCL backend occurs.
 */
class backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit backend_exception(const std::string &msg, source_location loc = source_location::current());
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] class_name the name of the thrown exception class
     * @param[in] loc the exception's call side information
     */
    explicit backend_exception(const std::string &msg, std::string_view class_name, source_location loc = source_location::current());
};

}  // namespace sycl

namespace adaptivecpp {

/**
 * @brief Exception type thrown if a problem with the AdaptiveCpp SYCL backend occurs.
 */
class backend_exception : public sycl::backend_exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::sycl::backend_exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit backend_exception(const std::string &msg, source_location loc = source_location::current());
};

}  // namespace adaptivecpp

namespace dpcpp {

/**
 * @brief Exception type thrown if a problem with the DPC++ SYCL backend occurs.
 */
class backend_exception : public sycl::backend_exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to plssvm::sycl::backend_exception.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit backend_exception(const std::string &msg, source_location loc = source_location::current());
};

}  // namespace dpcpp

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_SYCL_CSVM_HPP_
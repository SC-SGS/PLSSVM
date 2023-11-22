/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for testing the exception classes' functionality.
 */

#ifndef PLSSVM_TESTS_EXCEPTIONS_UTILITY_HPP_
#define PLSSVM_TESTS_EXCEPTIONS_UTILITY_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::{*_exception}

#include <string_view>  // std::string_view

/**
 * @def PLSSVM_CREATE_EXCEPTION_TYPE_NAME
 * @brief Defines a macro to create conversion functions from the exception types to their name as string representation.
 * @param[in] type the exception type to convert to a string
 */
#define PLSSVM_CREATE_EXCEPTION_TYPE_NAME(type) \
    template <>                                 \
    [[nodiscard]] constexpr inline std::string_view exception_type_name<type>() { return #type; }

namespace util {
// used that `exception_type_name` doesn't also print plssvm::
using namespace plssvm;

/**
 * @brief Tries to convert the given exception to its name as string representation.
 * @details The definition is marked as **deleted** if `T` isn't one of the custom PLSSVM exception types.
 * @tparam T the exception type to convert to a string
 * @return the name of the exception type `T` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr inline std::string_view exception_type_name() = delete;

// create exception type -> string mapping for all custom exception types
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(invalid_parameter_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(file_reader_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(data_set_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(file_not_found_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(invalid_file_format_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(unsupported_backend_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(unsupported_kernel_type_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(gpu_device_ptr_exception)
PLSSVM_CREATE_EXCEPTION_TYPE_NAME(matrix_exception)

}  // namespace util

#undef PLSSVM_CREATE_EXCEPTION_TYPE_NAME

#endif  // PLSSVM_TESTS_EXCEPTIONS_UTILITY_HPP_

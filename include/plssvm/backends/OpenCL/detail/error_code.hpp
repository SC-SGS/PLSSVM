/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Small wrapper around OpenCL's error codes.
 */

#pragma once

#include "CL/cl.h"        // cl_int, CL_SUCCESS
#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <iosfwd>       // std::ostream (forward declaration only)
#include <string_view>  // std::string_view

namespace plssvm::opencl::detail {

/**
 * @brief Class wrapping an OpenCL error code.
 * @details Inspired by [std::error_code](https://en.cppreference.com/w/cpp/error/error_code).
 */
class error_code {
  public:
    /**
     * @brief Construct a new error code indicating success (`CL_SUCCESS`).
     */
    error_code() = default;
    /**
     * @brief Construct a new error code wrapping the OpenCL error code @p err.
     * @param[in] err the OpenCL error code
     */
    error_code(cl_int err) noexcept;

    /**
     * @brief Assign an OpenCL error code to this.
     * @param[in] err the OpenCL error code
     * @return `*this`
     */
    error_code &operator=(cl_int err) noexcept;
    /**
     * @brief Assign an OpenCL error code to this.
     * @param[in] err the OpenCL error code
     */
    void assign(cl_int err) noexcept;

    /**
     * @brief Sets to error code value back to `CL_SUCCESS`.
     */
    void clear() noexcept;

    /**
     * @brief Obtain the value of the error code.
     * @return the error code value (`[[nodiscard]]`)
     */
    [[nodiscard]] cl_int value() const noexcept;
    /**
     * @brief Obtains the explanatory string of the error code.
     * @return the string representation of the error code
     */
    [[nodiscard]] std::string_view message() const noexcept;
    /**
     * @brief Checks whether the error code indicates success or not.
     * @return `true` if the error code is `CL_SUCCESS`, otherwise `false`
     */
    [[nodiscard]] explicit operator bool() const noexcept;
    /**
     * @brief Overloads the addressof operator to be able to set the wrapped error code value using an out-parameter
     *        in calls to OpenCL functions.
     * @return pointer to the wrapped OpenCL error code
     */
    [[nodiscard]] cl_int *operator&() noexcept;

  private:
    cl_int err_ = CL_SUCCESS;
};

/**
 * @brief Overload stream-insertion operator to print the error code and its message to the output.
 * @details Example output of an error code:
 * @code
 * "-1; CL_DEVICE_NOT_FOUND"
 * @endcode
 * @param[in,out] out the output-stream to write the kernel type to
 * @param[in] ec the error code
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, error_code ec);
/**
 * @brief Compares two error codes for equality.
 * @param[in] lhs the first error code
 * @param[in] rhs the second error code
 * @return `true` if both error codes are equal, `false` otherwise
 */
[[nodiscard]] bool operator==(error_code lhs, error_code rhs) noexcept;
/**
 * @brief Compares two error codes for inequality.
 * @param[in] lhs the first error code
 * @param[in] rhs the second error code
 * @return `true` if both error codes are unequal, `false` otherwise
 */
[[nodiscard]] bool operator!=(error_code lhs, error_code rhs) noexcept;

}  // namespace plssvm::opencl::detail
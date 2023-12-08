/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible SYCL implementations.
 */

#ifndef PLSSVM_BACKENDS_SYCL_IMPLEMENTATION_TYPE_HPP_
#define PLSSVM_BACKENDS_SYCL_IMPLEMENTATION_TYPE_HPP_
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm::sycl {

/**
 * @brief Enum class for all possible SYCL implementation types.
 */
enum class implementation_type {
    /** Use the available SYCL implementation. If more than one implementation is available, the macro PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION must be defined. */
    automatic,
    /** Use [DPC++](https://github.com/intel/llvm) as SYCL implementation. */
    dpcpp,
    /** Use [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (formerly known as hipSYCL) as SYCL implementation. */
    adaptivecpp
};

/**
 * @brief Return a list of all currently available SYCL implementations.
 * @details Only SYCL implementations that where found during the CMake configuration are available.
 * @return the available SYCL implementations (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<implementation_type> list_available_sycl_implementations();

/**
 * @brief Output the @p impl type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the backend type to
 * @param[in] impl the SYCL implementation type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, implementation_type impl);

/**
 * @brief Use the input-stream @p in to initialize the @p impl type.
 * @param[in,out] in input-stream to extract the backend type from
 * @param[in] impl the SYCL implementation type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, implementation_type &impl);

}  // namespace plssvm::sycl

template <>
struct fmt::formatter<plssvm::sycl::implementation_type> : fmt::ostream_formatter {};

#endif  // PLSSVM_BACKENDS_SYCL_IMPLEMENTATION_TYPE_HPP_
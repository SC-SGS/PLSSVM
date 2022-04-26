/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines all available kernel invoke types when using SYCL.
 */

#pragma once

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm::sycl_generic {

/**
 * @brief Enum class for all possible SYCL kernel invocation types.
 */
enum class kernel_invocation_type {
    /** Use the best kernel invocation type for the current SYCL implementation and target hardware platform. */
    automatic,
    /** Use the [*nd_range* invocation type](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_invoke). */
    nd_range,
    /** Use the SYCL specific [hierarchical invocation type](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_parallel_for_hierarchical_invoke). */
    hierarchical
};

/**
 * @brief Output the @p invocation type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the backend type to
 * @param[in] invocation the SYCL kernel invocation type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, kernel_invocation_type invocation);

/**
 * @brief Use the input-stream @p in to initialize the @p invocation type.
 * @param[in,out] in input-stream to extract the backend type from
 * @param[in] invocation the SYCL kernel invocation type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, kernel_invocation_type &invocation);

}  // namespace plssvm::sycl_generic

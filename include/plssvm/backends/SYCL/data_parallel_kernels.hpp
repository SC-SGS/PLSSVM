/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible SYCL data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DATA_PARALLEL_KERNELS_HPP_
#define PLSSVM_BACKENDS_SYCL_DATA_PARALLEL_KERNELS_HPP_
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm::sycl {

/**
 * @brief Enum class for all possible SYCL data parallel kernels.
 */
enum class data_parallel_kernel {
    /** Use the best data parallel kernel for the current SYCL implementation and target hardware platform. In practice, will nearly always map to work-group data parallel kernels. */
    automatic,
    /** Use the [`basic` data parallel kernels](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_basic_data_parallel_kernels). */
    basic,
    /** Use the [`work-group` data parallel kernels](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_work_group_data_parallel_kernels). */
    work_group,
    /** Use the [`hierarchical` data parallel kernels](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_hierarchical_data_parallel_kernels). **Note:** deprecated in newer SYCL version, will be replaced with a "better" version in future SYCL specifications. */
    hierarchical,
    /** Use the AdaptiveCpp specific [`scoped` parallelism](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/scoped-parallelism.md). */
    scoped
};

/**
 * @brief Return a list of all currently available SYCL data parallel kernels.
 * @details SYCL's hierarchical data parallel kernels and AdaptiveCpp's scoped parallelism can be disabled during the CMake configuration.
 * @return the available SYCL data parallel kernels (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<data_parallel_kernel> list_available_sycl_data_parallel_kernels();

/**
 * @brief Output the @p kernel_type type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the backend type to
 * @param[in] kernel_type the SYCL data parallel kernel
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, data_parallel_kernel kernel_type);

/**
 * @brief Use the input-stream @p in to initialize the @p kernel_type type.
 * @param[in,out] in input-stream to extract the backend type from
 * @param[in] kernel_type the SYCL data parallel kernel
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, data_parallel_kernel &kernel_type);

}  // namespace plssvm::sycl

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::sycl::data_parallel_kernel> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_BACKENDS_SYCL_DATA_PARALLEL_KERNELS_HPP_

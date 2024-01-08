/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions shared between all SYCL implementations.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_CONSTANTS_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_CONSTANTS_HPP_
#pragma once

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/constants.hpp"                             // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "sycl/sycl.hpp"  // sycl::range, sycl::nd_range

namespace plssvm::sycl::detail {

/**
 * @brief Calculate the SYCL execution range depending on the provided @p iteration_range and SYCL kernel @p invocation type.
 * @param[in] iteration_range the 2D range over which should be iterated
 * @param[in] invocation the SYCL kernel invocation type
 * @return the resulting execution range passed to SYCL's kernel call (`[[nodiscard]]`)
 */
[[nodiscard]] inline ::sycl::nd_range<2> calculate_execution_range(const ::sycl::range<2> iteration_range, const kernel_invocation_type invocation) {
    // the block size is the same across all kernels
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    // calculate the number of grids depending on the provided iteration range
    ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(iteration_range[0]) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))),
                           static_cast<std::size_t>(std::ceil(static_cast<double>(iteration_range[1]) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) };

    // for the nd_range kernels, the total number of work-items per dimension must be used
    if (invocation == kernel_invocation_type::nd_range) {
        grid[0] *= block[0];
        grid[1] *= block[1];
    }

    return ::sycl::nd_range<2>{ grid, block };
}

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_CONSTANTS_HPP_

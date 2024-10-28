/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a Kokkos function object for memsetting a device pointer with a specific value.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_KERNEL_DETAIL_MEMSET_KERNEL_HPP_
#define PLSSVM_BACKENDS_KOKKOS_KERNEL_DETAIL_MEMSET_KERNEL_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "Kokkos_Core.hpp"  // KOKKOS_INLINE_FUNCTION

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief A kernel to perform a memset-like operation on a Kokkos::View
 */
class device_memset_kernel {
  public:
    /**
     * @brief Memset all bytes in @p data to the provided @p pattern.
     * @param[out] data the array to memset
     * @param[in] pattern the memset pattern
     */
    device_memset_kernel(unsigned char* data, const unsigned char pattern) :
        data_{ data },
        pattern_{ pattern } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx the index representing the current point in the execution space
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t idx) const {
        data_[idx] = pattern_;
    }

  private:
    /// @cond Doxygen_suppress
    unsigned char* data_;
    const unsigned char pattern_;
    /// @endcond
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_KERNEL_DETAIL_MEMSET_KERNEL_HPP_

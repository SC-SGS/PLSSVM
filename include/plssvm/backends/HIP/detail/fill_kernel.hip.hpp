/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a HIP function for filling a device pointer with a specific value.
 */

#ifndef PLSSVM_BACKENDS_HIP_DETAIL_FILL_KERNEL_HPP_
#define PLSSVM_BACKENDS_HIP_DETAIL_FILL_KERNEL_HPP_
#pragma once

#include "hip/hip_runtime.h"      // HIP runtime functions
#include "hip/hip_runtime_api.h"  // HIP runtime functions

namespace plssvm::hip::detail {

/**
 * @brief Fill the array @p data with @p count values @p value starting at position @p pos.
 * @tparam value_type the type of the array
 * @tparam size_type  the unsigned size_type
 * @param[in,out] data the array to fill
 * @param[in] value the value to fill the array with
 * @param[in] pos the position at which to start te filling
 * @param[in] count the number of values to fill in the array starting at @p pos
 */
template <typename value_type, typename size_type>
__global__ void fill_array(value_type *data, value_type value, size_type pos, size_type count) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // fill the array
    if (idx < count) {
        data[pos + idx] = value;
    }
}

}  // namespace plssvm::hip::detail

#endif  // PLSSVM_BACKENDS_HIP_DETAIL_FILL_KERNEL_HPP_
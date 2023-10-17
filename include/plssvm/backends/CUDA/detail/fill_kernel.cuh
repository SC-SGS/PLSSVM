/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a CUDA function for filling a device pointer with a specific value.
 */

#ifndef PLSSVM_BACKENDS_CUDA_DETAIL_FILL_KERNEL_HPP_
#define PLSSVM_BACKENDS_CUDA_DETAIL_FILL_KERNEL_HPP_
#pragma once

namespace plssvm::cuda::detail {

/**
 * @brief Fill @p count values of the array @p data with the @p value starting at position @p start_pos.
 * @tparam value_type the type of the array and fill value
 * @tparam size_type the size type
 * @param[out] data the array to fill
 * @param[in] value the value to fill the array with
 * @param[in] start_pos the position to start filling the array
 * @param[in] count the number of values to fill
 */
template <typename value_type, typename size_type>
__global__ void fill_array(value_type *data, const value_type value, const size_type start_pos, const size_type count) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // fill the array
    if (idx < count) {
        data[start_pos + idx] = value;
    }
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_DETAIL_FILL_KERNEL_HPP_
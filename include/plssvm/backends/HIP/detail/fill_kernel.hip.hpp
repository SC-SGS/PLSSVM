/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_DETAIL_FILL_KERNEL_HPP_
#define PLSSVM_BACKENDS_HIP_DETAIL_FILL_KERNEL_HPP_
#pragma once

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip::detail {

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
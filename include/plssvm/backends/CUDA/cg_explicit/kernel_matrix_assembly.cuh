/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLE_CUH_
#define PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLE_CUH_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda {

template <typename real_type>
__global__ void device_kernel_assembly_linear(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features);

template <typename real_type>
__global__ void device_kernel_assembly_polynomial(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0);

template <typename real_type>
__global__ void device_kernel_assembly_rbf(const real_type *q, real_type *ret, const real_type *data_d, const real_type QA_cost, const real_type cost, const kernel_index_type num_rows, const kernel_index_type num_features, const real_type gamma);

}  // namespace plssvm::cuda

#endif  // PLSSVM_BACKENDS_CUDA_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLE_CUH_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "Kokkos_Core.hpp"  // TODO:

namespace plssvm::kokkos::detail {

class device_kernel_symm {
  public:
  private:
};

class device_kernel_symm_mirror {
  public:
  private:
};

class device_kernel_inplace_matrix_add {
  public:
  private:
};

class device_kernel_inplace_matrix_scale {
  public:
  private:
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_BLAS_HPP_

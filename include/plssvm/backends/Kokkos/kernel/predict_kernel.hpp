/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_KOKKOS_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"  // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type

#include "Kokkos_Core.hpp"  // TODO: Kokkos::atomic_add

namespace plssvm::kokkos::detail {

class device_kernel_w_linear {
  public:
  private:
};

class device_kernel_predict_linear {
  public:
  private:
};

template <kernel_function_type kernel_function, typename... Args>
class device_kernel_predict {
  public:
  private:
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_PREDICT_KERNEL_HPP_

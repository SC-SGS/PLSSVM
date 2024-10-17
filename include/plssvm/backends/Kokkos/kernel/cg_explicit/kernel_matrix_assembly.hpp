/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/backends/Kokkos/kernel/kernel_functions.hpp"  // plssvm::kokkos::detail::{feature_reduce, apply_kernel_function}
#include "plssvm/constants.hpp"                                // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/kernel_function_types.hpp"                    // plssvm::kernel_function_type

#include "Kokkos_Core.hpp"  // TODO:

namespace plssvm::kokkos::detail {

template <kernel_function_type kernel_function, typename... Args>
class device_kernel_assembly {
  public:
  private:
};

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_

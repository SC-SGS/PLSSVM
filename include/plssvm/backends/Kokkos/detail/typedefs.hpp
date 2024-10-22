/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few convenient Kokkos::View typedefs.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_TYPEDEFS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_TYPEDEFS_HPP_
#pragma once

#include "Kokkos_Core.hpp"  // Kokkos::View, Kokkos::DefaultExecutionSpace, Kokkos::HostSpace, Kokkos::MemoryUnmanaged

namespace plssvm::kokkos::detail {

/**
 * @brief Typedef for a simple Kokkos::View targeting the Kokkos::DefaultExecutionSpace.
 * @tparam T the type of the view's data
 */
template <typename T>
using device_view_type = Kokkos::View<T *, Kokkos::DefaultExecutionSpace>;

/**
 * @brief Typedef for a simple Kokkos::View always targeting the Kokkos::HostSpace.
 * @tparam T the type of the view's data
 */
template <typename T>
using host_view_type = Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_TYPEDEFS_HPP_

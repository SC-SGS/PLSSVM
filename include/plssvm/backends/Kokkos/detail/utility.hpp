/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_
#pragma once

#include "plssvm/backends/Kokkos/detail/execution_space.hpp"  // plssvm::kokkos::detail::execution_space
#include "plssvm/target_platforms.hpp"                        // plssvm::target_platform

#include "Kokkos_Core.hpp"  // TODO: ?

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace plssvm::kokkos::detail {

template <typename ExecSpace>
[[nodiscard]] execution_space determine_execution_space() noexcept {
    // determine the execution_space enumeration value based on the provided Kokkos execution space
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        return execution_space::cuda;
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        return execution_space::hip;
    }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::SYCL>) {
        return execution_space::sycl;
    }
#endif
#if defined(KOKKOS_ENABLE_HPX)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Experimental::HPX>) {
        return execution_space::hpx;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return execution_space::openmp;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMPTarget>) {
        return execution_space::openmp_target;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Experimental::OpenACC>) {
        return execution_space::openacc;
    }
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Threads>) {
        return execution_space::threads;
    }
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return execution_space::serial;
    }
#endif
}

[[nodiscard]] target_platform determine_default_target_platform_from_execution_space(execution_space space);

void check_execution_space_target_platform_combination(execution_space space, target_platform target);

[[nodiscard]] std::string get_device_name(execution_space space, std::size_t device_id);

void device_synchronize_all();

[[nodiscard]] std::string get_kokkos_version();

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_

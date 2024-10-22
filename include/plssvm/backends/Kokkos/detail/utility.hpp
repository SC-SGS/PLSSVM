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

#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"  // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::execution_space
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform

#include "Kokkos_Core.hpp"  // TODO: ?

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace plssvm::kokkos::detail {

[[nodiscard]] target_platform determine_default_target_platform_from_execution_space(execution_space space);

void check_execution_space_target_platform_combination(execution_space space, target_platform target);

template <typename ExecSpace>
[[nodiscard]] inline std::string get_device_name(const execution_space space, [[maybe_unused]] const ExecSpace &exec) {
    // TODO: implement for other backends!
    switch (space) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                return std::string{ exec.cuda_device_prop().name };
            });
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                return std::string{ exec.hip_device_prop().name };
            });
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                return exec.sycl_queue.get_device().get_info<sycl::info::device::name>();
            });
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            return "CPU host device";
        case execution_space::openmp_target:
            return "OpenMP target device";
        case execution_space::openacc:
            return "OpenACC target device";
    }
    return "unknown";
}

void device_synchronize(const Kokkos::DefaultExecutionSpace &exec);

[[nodiscard]] std::string get_kokkos_version();

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_

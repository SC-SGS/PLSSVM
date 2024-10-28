/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/utility.hpp"

#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"  // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"         // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::execution_space
#include "plssvm/detail/assert.hpp"                                 // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"                         // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"                                // plssvm::detail::contains
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform

#include "Kokkos_Core.hpp"    // Kokkos::ExecutionSpace, Kokkos::Impl::ManageStream
#include "Kokkos_Macros.hpp"  // Kokkos macros

#include "fmt/core.h"  // fmt::format

#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

namespace plssvm::kokkos::detail {

std::map<target_platform, std::vector<execution_space>> available_target_platform_to_execution_space_mapping() {
    std::map<target_platform, std::vector<execution_space>> available_map{};

    // TODO: only return really POSSIBLE target platforms?
    // iterate over all available execution spaces
    for (const execution_space space : list_available_execution_spaces()) {
        switch (space) {
            case execution_space::cuda:
                // NVIDIA GPUs only
                available_map[target_platform::gpu_nvidia].push_back(execution_space::cuda);
                break;
            case execution_space::hip:
                // NVIDIA and AMD GPUs possible
                available_map[target_platform::gpu_nvidia].push_back(execution_space::hip);
                available_map[target_platform::gpu_amd].push_back(execution_space::hip);
                break;
            case execution_space::sycl:
            case execution_space::openacc:
                // all GPUs and CPU possible
                available_map[target_platform::gpu_nvidia].push_back(execution_space::sycl);
                available_map[target_platform::gpu_amd].push_back(execution_space::sycl);
                available_map[target_platform::gpu_intel].push_back(execution_space::sycl);
                available_map[target_platform::cpu].push_back(execution_space::sycl);
                break;
            case execution_space::openmp_target:
                // all GPUs
                available_map[target_platform::gpu_nvidia].push_back(execution_space::openmp_target);
                available_map[target_platform::gpu_amd].push_back(execution_space::openmp_target);
                available_map[target_platform::gpu_intel].push_back(execution_space::openmp_target);
                break;
            case execution_space::hpx:
            case execution_space::openmp:
            case execution_space::threads:
            case execution_space::serial:
                // all these execution spaces are CPU only
                available_map[target_platform::cpu].push_back(space);
                break;
        }
    }

    // the map must at least have one entry
    PLSSVM_ASSERT(!available_map.empty(), "At least one target platform must be available!");
    // the automatic target platform must not be present
    PLSSVM_ASSERT(!::plssvm::detail::contains(available_map, target_platform::automatic), "The automatic target platform may not be present!");

    return available_map;
}

std::string get_device_name([[maybe_unused]] const device_wrapper &dev) {
    switch (dev.get_execution_space()) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                return std::string{ dev.get<execution_space::cuda>().cuda_device_prop().name };
            });
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                return std::string{ dev.get<execution_space::hip>().hip_device_prop().name };
            });
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                return dev.get<execution_space::sycl>().sycl_queue.get_device().get_info<sycl::info::device::name>();
            });
        case execution_space::hpx:
            return "HPX CPU host device";
        case execution_space::openmp:
            return "OpenMP CPU host device";
        case execution_space::openmp_target:
            // TODO: device name?
            return "OpenMP target device";
        case execution_space::openacc:
            // TODO: device name?
            return "OpenACC target device";
        case execution_space::threads:
            return "std::threads CPU host device";
        case execution_space::serial:
            return "serial CPU host device";
    }
    return "unknown";
}

void device_synchronize(const device_wrapper &dev) {
    dev.execute([](const auto &device) {
        device.fence();
    });
}

std::string get_kokkos_version() {
    // get the Kokkos version
    return fmt::format("{}.{}.{}", KOKKOS_VERSION_MAJOR, KOKKOS_VERSION_MINOR, KOKKOS_VERSION_PATCH);
}

}  // namespace plssvm::kokkos::detail

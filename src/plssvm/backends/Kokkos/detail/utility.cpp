/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/utility.hpp"

#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"  // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/exceptions.hpp"                    // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::execution_space
#include "plssvm/detail/assert.hpp"                                 // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"                                // plssvm::detail::unreachable
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform

#include "Kokkos_Core.hpp"    // Kokkos::DefaultExecutionSpace, Kokkos::num_devices, Kokkos::Cuda, Kokkos::Hip, Kokkos::Sycl, Kokkos::Impl::ManageStream
#include "Kokkos_Macros.hpp"  // Kokkos macros

#include "fmt/core.h"  // fmt::format

#include <string>  // std::string
#include <vector>  // std::vector

namespace plssvm::kokkos::detail {

target_platform determine_default_target_platform_from_execution_space(const execution_space space) {
    switch (space) {
        case execution_space::cuda:
            return target_platform::gpu_nvidia;
        case execution_space::hip:
            return target_platform::gpu_amd;  // TODO: or gpu_nvidia :/
        case execution_space::sycl:
        case execution_space::openmp_target:
        case execution_space::openacc:
            return target_platform::gpu_nvidia;  // TODO: what to return here?
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            return target_platform::cpu;
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

void check_execution_space_target_platform_combination(const execution_space space, const target_platform target) {
    PLSSVM_ASSERT(target != target_platform::automatic, "The provided target platform may not be the automatic target platform!");

    switch (space) {
        case execution_space::cuda:
            if (target != target_platform::gpu_nvidia) {
                throw backend_exception{ fmt::format("The target platform {} is not supported for Kokkos {} execution space!", target, space) };
            }
            break;
        case execution_space::hip:
            if (target != target_platform::gpu_amd && target != target_platform::gpu_nvidia) {
                throw backend_exception{ fmt::format("The target platform {} is not supported for Kokkos {} execution space!", target, space) };
            }
            break;
        case execution_space::sycl:
            // SYCL may support all target platforms!
            // TODO: use SYCL specific functions to check?
        case execution_space::openmp_target:
            // OpenMP Target Offloading may support all target platforms!
            // TODO: use OpenMP Target Offloading specific functions to check?
        case execution_space::openacc:
            // OpenACC may support all target platforms!
            // TODO: use OpenACC Target Offloading specific functions to check?
            break;
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            if (target != target_platform::cpu) {
                throw backend_exception{ fmt::format("The target platform {} is not supported for Kokkos {} execution space!", target, space) };
            }
            break;
    }
}

std::vector<Kokkos::DefaultExecutionSpace> get_device_list(const execution_space space, [[maybe_unused]] const target_platform target) {
    std::vector<Kokkos::DefaultExecutionSpace> devices{};
    switch (space) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                for (int device = 0; device < Kokkos::num_devices(); ++device) {
                    // create CUDA stream using the CUDA specific functions
                    cudaSetDevice(device);
                    cudaStream_t stream{};
                    cudaStreamCreate(&stream);
                    // create Kokkos execution space for the specific device
                    // Note: it is important to pass the cudaStream_t lifetime to be managed by Kokkos
                    devices.emplace_back(Kokkos::Cuda(stream, Kokkos::Impl::ManageStream::yes));
                }
                return devices;
            });
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                for (int device = 0; device < Kokkos::num_devices(); ++device) {
                    // HIP CUDA stream using the HIP specific functions
                    hipSetDevice(device);
                    hipStream_t stream{};
                    hipStreamCreate(&stream);
                    // create Kokkos execution space for the specific device
                    // Note: it is important to pass the hipStream_t lifetime to be managed by Kokkos
                    devices.emplace_back(Kokkos::Hip(stream, Kokkos::Impl::ManageStream::yes));
                }
                return devices;
            });
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                // TODO: use all available devices -> not that trivial
                // TODO: handle target
                devices.emplace_back(Kokkos::SYCL{});
                return devices;
            });
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            devices.emplace_back(Kokkos::DefaultExecutionSpace{});
            return devices;
        case execution_space::openmp_target:
        case execution_space::openacc:
            // TODO: implement
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space) };
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

std::string get_device_name(const execution_space space, [[maybe_unused]] const Kokkos::DefaultExecutionSpace &exec) {
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

void device_synchronize(const Kokkos::DefaultExecutionSpace &exec) {
    exec.fence();
}

std::string get_kokkos_version() {
    // get the Kokkos version
    return fmt::format("{}.{}.{}", KOKKOS_VERSION_MAJOR, KOKKOS_VERSION_MINOR, KOKKOS_VERSION_PATCH);
}

}  // namespace plssvm::kokkos::detail

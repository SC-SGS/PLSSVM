/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/utility.hpp"

#include "plssvm/backends/Kokkos/detail/execution_space.hpp"  // plssvm::kokkos::detail::execution_space
#include "plssvm/backends/Kokkos/exceptions.hpp"              // plssvm::kokkos::backend_exception
#include "plssvm/detail/assert.hpp"                           // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"                          // plssvm::detail::unreachable
#include "plssvm/target_platforms.hpp"                        // plssvm::target_platform

#include "Kokkos_Macros.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
    #include "cuda_runtime.h"  // cudaDeviceProp, cudaGetDeviceProperties
#endif
#if defined(KOKKOS_ENABLE_HIP)
    #include "hip/hip_runtime_api.h"  // HIP runtime functions
#endif

#include "fmt/core.h"  // fmt::format

#include <cstddef>  // std::size_t
#include <string>   // std::string

namespace plssvm::kokkos::detail {

target_platform determine_default_target_platform_from_execution_space(const execution_space space) {
    switch (space) {
        case execution_space::cuda:
            return target_platform::gpu_nvidia;
        case execution_space::hip:
            return target_platform::gpu_amd;
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
            if (target != target_platform::gpu_amd) {
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

// TODO: error checks?

std::string get_device_name(const execution_space space, const std::size_t device_id) {
    // TODO: implement for other backends!
    switch (space) {
        case execution_space::cuda:
#if defined(KOKKOS_ENABLE_CUDA)
            {
                cudaDeviceProp prop{};
                cudaGetDeviceProperties(&prop, static_cast<int>(device_id));
                return std::string{ prop.name };
            }
#else
            throw backend_exception{ fmt::format("Unsupported Kokkos execution space \"{}\"!", space) };
#endif
        case execution_space::hip:
#if defined(KOKKOS_ENABLE_HIP)
            {
                hipDeviceProp_t prop{};
                hipGetDeviceProperties(&prop, static_cast<int>(device_id));
                return std::string{ prop.name };
            }
#else
            throw backend_exception{ fmt::format("Unsupported Kokkos execution space \"{}\"!", space) };
#endif
        case execution_space::openmp:
#if defined(KOKKOS_ENABLE_HIP)
            return "CPU host device";
#else
            throw backend_exception{ fmt::format("Unsupported Kokkos execution space \"{}\"!", space) };
#endif
        case execution_space::sycl:
        case execution_space::hpx:
        case execution_space::openmp_target:
        case execution_space::openacc:
        case execution_space::threads:
        case execution_space::serial:
            throw backend_exception{ fmt::format("Unsupported Kokkos execution space \"{}\"!", space) };
    }
    return "unknown";
}

void device_synchronize(const Kokkos::DefaultExecutionSpace& exec) {
    exec.fence();
}

std::string get_kokkos_version() {
    // get the Kokkos version
    return fmt::format("{}.{}.{}", KOKKOS_VERSION_MAJOR, KOKKOS_VERSION_MINOR, KOKKOS_VERSION_PATCH);
}

}  // namespace plssvm::kokkos::detail

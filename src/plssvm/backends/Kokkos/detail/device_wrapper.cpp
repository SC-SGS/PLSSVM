/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"

#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"  // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::execution_space
#include "plssvm/detail/logging_without_performance_tracking.hpp"   // plssvm::detail::log_untracked
#include "plssvm/detail/utility.hpp"                                // plssvm::detail::unreachable
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                              // plssvm::verbosity_level

#include "Kokkos_Core.hpp"  // Kokkos::num_devices, Kokkos::ExecutionSpace

#include <vector>  // std::vector

namespace plssvm::kokkos::detail {

std::vector<device_wrapper> get_device_list(const execution_space space, [[maybe_unused]] const target_platform target) {
    std::vector<device_wrapper> devices{};
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
            });
            break;
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                for (int device = 0; device < Kokkos::num_devices(); ++device) {
                    // HIP CUDA stream using the HIP specific functions
                    hipSetDevice(device);
                    hipStream_t stream{};
                    hipStreamCreate(&stream);
                    // create Kokkos execution space for the specific device
                    // Note: it is important to pass the hipStream_t lifetime to be managed by Kokkos
                    devices.emplace_back(Kokkos::HIP(stream, Kokkos::Impl::ManageStream::yes));
                }
            });
            break;
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                // TODO: use all available devices -> not that trivial
                // TODO: handle target <- if provide queue -> managed?
                devices.emplace_back(Kokkos::SYCL{});
            });
            break;
        case execution_space::hpx:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HPX([&]() {
                devices.emplace_back(Kokkos::Experimental::HPX{});
            });
            break;
        case execution_space::openmp:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMP([&]() {
                // Note: if OpenMP should be used as device  must be set in order for it to work!
                if (omp_get_nested() == 0) {
                    ::plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                                                    "WARNING: In order for Kokkos::OpenMP to work properly, we have to set \"omp_set_nested(1)\"!\n");
                    // enable OMP_NESTED support
                    // Note: function is officially deprecated but still necessary for Kokkos::OpenMP to work properly
                    omp_set_nested(1);
                }
                devices.emplace_back(Kokkos::OpenMP{});
            });
            break;
        case execution_space::openmp_target:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMPTARGET([&]() {
                // TODO: multi-GPU?
                devices.emplace_back(Kokkos::OpenMPTarget{});
            });
            break;
        case execution_space::openacc:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC([&]() {
                // TODO: multi-GPU?
                devices.emplace_back(Kokkos::OpenACC{});
            });
            break;
        case execution_space::threads:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_THREADS([&]() {
                devices.emplace_back(Kokkos::Threads{});
            });
            break;
        case execution_space::serial:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SERIAL([&]() {
                devices.emplace_back(Kokkos::Serial{});
            });
            break;
    }
    return devices;
}

}  // namespace plssvm::kokkos::detail

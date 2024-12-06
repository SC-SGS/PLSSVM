/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"

#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"  // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/exceptions.hpp"                    // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::execution_space
#include "plssvm/detail/assert.hpp"                                 // PLSSVM_ASSERT
#include "plssvm/detail/logging_without_performance_tracking.hpp"   // plssvm::detail::log_untracked
#include "plssvm/detail/string_utility.hpp"                         // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"                                // plssvm::detail::contains
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                              // plssvm::verbosity_level

#include "Kokkos_Core.hpp"  // Kokkos::num_devices, Kokkos::ExecutionSpace

#include <vector>  // std::vector

#if defined(KOKKOS_ENABLE_CUDA)
    #define PLSSVM_CUDA_ERROR_CHECK(err)                                                                                                            \
        if ((err) != cudaSuccess) {                                                                                                                 \
            throw plssvm::kokkos::backend_exception{ fmt::format("Kokkos::Cuda assert '{}': {}", cudaGetErrorName(err), cudaGetErrorString(err)) }; \
        }
#endif

#if defined(KOKKOS_ENABLE_HIP)
    #define PLSSVM_HIP_ERROR_CHECK(err)                                                                                                  \
        if ((err) != hipSuccess) {                                                                                                       \
            throw plssvm::kokkos::backend_exception{ fmt::format("HIP assert '{}': {}", hipGetErrorName(err), hipGetErrorString(err)) }; \
        }
#endif

namespace plssvm::kokkos::detail {

std::vector<device_wrapper> get_device_list(const execution_space space, [[maybe_unused]] const target_platform target) {
    PLSSVM_ASSERT(space != execution_space::automatic, "The automatic execution_space may not be provided to this function!");

    std::vector<device_wrapper> devices{};
    switch (space) {
        case execution_space::automatic:
            throw backend_exception{ "Unsupported execution_space::automatic provided!" };
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                for (int device = 0; device < Kokkos::num_devices(); ++device) {
                    // create CUDA stream using the CUDA specific functions
                    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device));
                    cudaStream_t stream{};
                    PLSSVM_CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
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
                    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device));
                    hipStream_t stream{};
                    PLSSVM_HIP_ERROR_CHECK(hipStreamCreate(&stream));
                    // create Kokkos execution space for the specific device
                    // Note: it is important to pass the hipStream_t lifetime to be managed by Kokkos
                    devices.emplace_back(Kokkos::HIP(stream, Kokkos::Impl::ManageStream::yes));
                }
            });
            break;
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL(([&]() {
                // all user provided sycl::queues must be in-order queues
                ::sycl::property_list props{ ::sycl::property::queue::in_order{} };

                for (const auto &platform : ::sycl::platform::get_platforms()) {
                    for (const auto &device : platform.get_devices()) {
                        // Note: Kokkos is IntelLLVM/DPC++/icpx only
                        if (device.is_cpu() && target == target_platform::cpu) {
                            devices.emplace_back(Kokkos::SYCL{ ::sycl::queue{ device, props } });
                        } else if (device.is_gpu()) {
                            // the current device is a GPU
                            // get vendor string and convert it to all lower case
                            const std::string vendor_string = ::plssvm::detail::as_lower_case(device.get_info<::sycl::info::device::vendor>());
                            // get platform name of current GPU device and convert it to all lower case
                            const std::string platform_string = ::plssvm::detail::as_lower_case(platform.get_info<::sycl::info::platform::name>());

                            // check vendor string and insert to correct target platform
                            if (::plssvm::detail::contains(vendor_string, "nvidia") && target == target_platform::gpu_nvidia) {
                                devices.emplace_back(Kokkos::SYCL{ ::sycl::queue{ device, props } });
                            } else if ((::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices")) && target == target_platform::gpu_amd) {
                                devices.emplace_back(Kokkos::SYCL{ ::sycl::queue{ device, props } });
                            } else if (::plssvm::detail::contains(vendor_string, "intel") && target == target_platform::gpu_intel) {
                                devices.emplace_back(Kokkos::SYCL{ ::sycl::queue{ device, props } });
                            }
                        }
                    }
                }
            }));
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
                // TODO: implement multi-GPU support?
                devices.emplace_back(Kokkos::Experimental::OpenMPTarget{});
            });
            break;
        case execution_space::openacc:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC([&]() {
                // TODO: implement multi-GPU support?
                devices.emplace_back(Kokkos::Experimental::OpenACC{});
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

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/hardware_sampler.hpp"

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    #include "plssvm/detail/tracking/cpu/hardware_sampler.hpp"  // plssvm::detail::tracking::cpu_hardware_sampler
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
    #include "plssvm/detail/tracking/gpu_nvidia/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_nvidia_hardware_sampler
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
    #include "plssvm/detail/tracking/gpu_amd/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_amd_hardware_sampler
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
    #include "plssvm/detail/tracking/gpu_intel/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_intel_hardware_sampler
#endif

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"         // plssvm::detail::unreachable
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr, std::make_unique
#include <vector>   // std::vector

using namespace std::chrono_literals;

namespace plssvm::detail::tracking {

std::unique_ptr<hardware_sampler> make_hardware_sampler([[maybe_unused]] const target_platform target, [[maybe_unused]] const std::size_t device_id, [[maybe_unused]] const std::chrono::milliseconds sampling_interval) {
    switch (target) {
        case target_platform::automatic:
            return make_hardware_sampler(determine_default_target_platform(), device_id, sampling_interval);
        case target_platform::cpu:
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
            return std::make_unique<cpu_hardware_sampler>(sampling_interval);
#else
            throw hardware_sampling_exception{ "Hardware sampling on CPUs wasn't enabled!" };
#endif
        case target_platform::gpu_nvidia:
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
            return std::make_unique<gpu_nvidia_hardware_sampler>(device_id, sampling_interval);
#else
            throw hardware_sampling_exception{ "Provided 'gpu_nvidia' as target_platform, but hardware sampling on NVIDIA GPUs using NVML wasn't enabled! Try setting an nvidia target during CMake configuration." };
#endif
        case target_platform::gpu_amd:

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
            return std::make_unique<gpu_amd_hardware_sampler>(device_id, sampling_interval);
#else
            throw hardware_sampling_exception{ "Provided 'gpu_amd' as target_platform, but hardware sampling on AMD GPUs using ROCm SMI wasn't enabled! Try setting an amd target during CMake configuration." };
#endif
        case target_platform::gpu_intel:
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
            return std::make_unique<gpu_intel_hardware_sampler>(device_id, sampling_interval);
#else
            throw hardware_sampling_exception{ "Provided 'gpu_intel' as target_platform, but hardware sampling on Intel GPUs using Level Zero wasn't enabled! Try setting an intel target during CMake configuration." };
#endif
    }

    detail::unreachable();
}

std::vector<std::unique_ptr<hardware_sampler>> create_hardware_sampler(const target_platform target, const std::size_t num_devices, const std::chrono::milliseconds sampling_interval) {
    if (num_devices == 0) {
        throw hardware_sampling_exception{ "The number of devices must be greater than 0!" };
    }

    std::vector<std::unique_ptr<hardware_sampler>> sampler{};

    // ignore the CPU target platform since we ALWAYS add a CPU hardware sampler if possible
    if ((target != target_platform::automatic && target != target_platform::cpu) || (target == target_platform::automatic && determine_default_target_platform() != target_platform::cpu)) {
        for (std::size_t device = 0; device < num_devices; ++device) {
            sampler.push_back(make_hardware_sampler(target, device, sampling_interval));
        }
    }

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    // if available, ALWAYS add a CPU hardware sampler
    sampler.push_back(make_hardware_sampler(target_platform::cpu, 0, sampling_interval));
#endif

    return sampler;
}

}  // namespace plssvm::detail::tracking

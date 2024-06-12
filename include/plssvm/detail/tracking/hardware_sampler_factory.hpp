/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all hardware samplers.
 */

#ifndef PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_
#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_

#include "plssvm/detail/tracking/hardware_sampler.hpp"

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    #include "plssvm/detail/tracking/turbostat_hardware_sampler.hpp"  // plssvm::detail::tracking::turbostat_hardware_sampler
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_NVML_ENABLED)
    #include "plssvm/detail/tracking/nvml_hardware_sampler.hpp"  // plssvm::detail::tracking::nvml_hardware_sampler
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_ROCM_SMI_ENABLED)
    #include "plssvm/detail/tracking/rocm_smi_hardware_sampler.hpp"  // plssvm::detail::tracking::rocm_smi_hardware_sampler
#endif

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"         // plssvm::detail::unreachable
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr, std::make_unique
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

[[nodiscard]] inline std::unique_ptr<hardware_sampler> make_hardware_sampler([[maybe_unused]] const target_platform target, [[maybe_unused]] const std::size_t device_id, [[maybe_unused]] const std::chrono::milliseconds sampling_interval) {
    switch (target) {
        case target_platform::automatic:
            return make_hardware_sampler(determine_default_target_platform(), device_id, sampling_interval);
        case target_platform::cpu:
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
            return std::make_unique<turbostat_hardware_sampler>(sampling_interval);
#else
            throw hardware_sampling_exception{ "Hardware sampling on CPUs wasn't enabled!" };
#endif
        case target_platform::gpu_nvidia:
#if defined(PLSSVM_HARDWARE_TRACKING_VIA_NVML_ENABLED)
            return std::make_unique<nvml_hardware_sampler>(device_id, sampling_interval);
#else
            throw hardware_sampling_exception{ "Provided 'gpu_nvidia' as target_platform, but hardware sampling on NVIDIA GPUs using NVML wasn't enabled! Try setting an nvidia target during CMake configuration." };
#endif
        case target_platform::gpu_amd:

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_ROCM_SMI_ENABLED)
            return std::make_unique<rocm_smi_hardware_sampler>(device_id, sampling_interval);
#else
            throw hardware_sampling_exception{ "Provided 'gpu_amd' as target_platform, but hardware sampling on AMD GPUs using ROCm SMI wasn't enabled! Try setting an amd target during CMake configuration." };
#endif
        case target_platform::gpu_intel:
            throw hardware_sampling_exception{ "Intel GPU hardware sampling currently not implemented!" };  // TODO: implement
    }

    detail::unreachable();
}

class global_hardware_sampler {
    friend std::unique_ptr<global_hardware_sampler> std::make_unique<global_hardware_sampler>(const plssvm::target_platform &, const std::size_t &, const std::chrono::milliseconds &);

  public:
    static void init_instance(const target_platform target, const std::size_t num_devices, const std::chrono::milliseconds sampling_interval) {
        PLSSVM_ASSERT(instance_ == nullptr, "Instance already initialized!");
        if (instance_ == nullptr) {
            instance_ = std::make_unique<global_hardware_sampler>(target, num_devices, sampling_interval);
        }
    }

    [[nodiscard]] static std::unique_ptr<global_hardware_sampler> &get_instance_ptr() noexcept {
        return instance_;
    }

    [[nodiscard]] static global_hardware_sampler &get_instance() noexcept {
        PLSSVM_ASSERT(instance_ != nullptr, "Can't get an instance from a nullptr! Maybe a call to 'init_instance' is missing?");
        return *instance_;
    }

    [[nodiscard]] std::vector<std::unique_ptr<hardware_sampler>> &get_sampler() noexcept { return sampler_; }

    template <typename Func, typename... Args>
    static void for_each_sampler(Func f, Args... args) {
        for (std::unique_ptr<hardware_sampler> &sampler : get_instance().get_sampler()) {
            ((*sampler).*f)(args...);
        }
    }

  private:
    global_hardware_sampler(const target_platform target, const std::size_t num_devices, const std::chrono::milliseconds sampling_interval) {
        // ignore the CPU target platform since we ALWAYS add a CPU hardware sampler if possible
        if (target != target_platform::cpu) {
            for (std::size_t device = 0; device < num_devices; ++device) {
                sampler_.push_back(make_hardware_sampler(target, device, sampling_interval));
            }
        }

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
        // if available, ALWAYS add a CPU hardware sampler
        sampler_.push_back(make_hardware_sampler(target_platform::cpu, 0, sampling_interval));
#endif
    }

    std::vector<std::unique_ptr<hardware_sampler>> sampler_{};

    inline static std::unique_ptr<global_hardware_sampler> instance_{};
};

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_INIT(target, num_devices) \
        plssvm::detail::tracking::global_hardware_sampler::init_instance(target, num_devices, PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_START_SAMPLING() \
        plssvm::detail::tracking::global_hardware_sampler::for_each_sampler(&plssvm::detail::tracking::hardware_sampler::start_sampling);

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_STOP_SAMPLING() \
        plssvm::detail::tracking::global_hardware_sampler::for_each_sampler(&plssvm::detail::tracking::hardware_sampler::stop_sampling);

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_PAUSE_SAMPLING() \
        plssvm::detail::tracking::global_hardware_sampler::for_each_sampler(&plssvm::detail::tracking::hardware_sampler::pause_sampling);

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_RESUME_SAMPLING() \
        plssvm::detail::tracking::global_hardware_sampler::for_each_sampler(&plssvm::detail::tracking::hardware_sampler::resume_sampling);

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_ADD_EVENT(name) \
        plssvm::detail::tracking::global_hardware_sampler::for_each_sampler(&plssvm::detail::tracking::hardware_sampler::add_event, name);

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_CLEANUP() \
        plssvm::detail::tracking::global_hardware_sampler::get_instance_ptr().reset();

#else

    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_INIT(target, num_devices)
    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_START_SAMPLING()
    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_STOP_SAMPLING()
    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_PAUSE_SAMPLING()
    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_RESUME_SAMPLING()
    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_ADD_EVENT(name)
    #define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_CLEANUP()

#endif

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_

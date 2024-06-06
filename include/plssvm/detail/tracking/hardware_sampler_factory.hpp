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
#include "plssvm/detail/tracking/nvml_hardware_sampler.hpp"  // TODO: guard behind ifdef
#include "plssvm/target_platforms.hpp"

#include <chrono>  // std::chrono::milliseconds, std::chrono_literals namespace
#include <memory>  // std::unique_ptr, std::make_unique
#include <vector>  // std::vector

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

// TODO: template parameter
inline std::unique_ptr<nvml_hardware_sampler> hardware_sampler_factory(const target_platform target, const std::size_t device_id, const std::chrono::milliseconds sampling_interval = 100ms) {
    // TODO: may not be automatic

    switch (target) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
            return nullptr;  // TODO: implement
        case target_platform::gpu_nvidia:
            return std::make_unique<nvml_hardware_sampler>(device_id, sampling_interval);
        case target_platform::gpu_amd:
            return nullptr;  // TODO: implement
        case target_platform::gpu_intel:
            return nullptr;  // TODO: implement
    }

    return nullptr;  // TODO:
}

// TODO: better
inline std::vector<std::unique_ptr<nvml_hardware_sampler>> &global_hardware_sampler(const target_platform target = target_platform::automatic, const std::size_t num_devices = 0, const std::chrono::milliseconds sampling_interval = 100ms) {
    static std::vector<std::unique_ptr<nvml_hardware_sampler>> sampler = [=]() {
        std::vector<std::unique_ptr<nvml_hardware_sampler>> s{};
        for (std::size_t device = 0; device < num_devices; ++device) {
            s.push_back(hardware_sampler_factory(target, device, sampling_interval));
        }
        return s;
    }();
    return sampler;
}

#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_INIT(target, num_devices) \
    plssvm::detail::tracking::global_hardware_sampler(target, num_devices, PLSSVM_HARDWARE_SAMPLING_INTERVAL);

#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_START_SAMPLING()                \
    for (const auto &s : plssvm::detail::tracking::global_hardware_sampler()) { \
        s->start_sampling();                                                    \
    }

#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_STOP_SAMPLING()                 \
    for (const auto &s : plssvm::detail::tracking::global_hardware_sampler()) { \
        s->stop_sampling();                                                     \
    }

#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_ADD_EVENT(name)                 \
    for (const auto &s : plssvm::detail::tracking::global_hardware_sampler()) { \
        s->add_event(name);                                                     \
    }

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_

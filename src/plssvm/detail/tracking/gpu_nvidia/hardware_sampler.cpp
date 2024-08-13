/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_nvidia/hardware_sampler.hpp"

#include "plssvm/detail/tracking/gpu_nvidia/nvml_device_handle_impl.hpp"  // plssvm::detail::tracking::nvml_device_handle implementation
#include "plssvm/detail/tracking/gpu_nvidia/nvml_samples.hpp"             // plssvm::detail::tracking::{nvml_general_samples, nvml_clock_samples, nvml_power_samples, nvml_memory_samples, nvml_temperature_samples}
#include "plssvm/detail/tracking/gpu_nvidia/utility.hpp"                  // PLSSVM_NVML_ERROR_CHECK
#include "plssvm/detail/tracking/hardware_sampler.hpp"                    // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/performance_tracker.hpp"                 // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/tracking/utility.hpp"                             // plssvm::detail::tracking::{durations_from_reference_time, time_points_to_epoch}
#include "plssvm/exceptions/exceptions.hpp"                               // plssvm::exception, plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"                                    // plssvm::target_platform

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join
#include "nvml.h"        // NVML runtime functions

#include <algorithm>  // std::min_element
#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <exception>  // std::exception, std::terminate
#include <ios>        // std::ios_base
#include <iostream>   // std::cerr, std::endl
#include <optional>   // std::optional
#include <ostream>    // std::ostream
#include <string>     // std::string
#include <thread>     // std::this_thread
#include <vector>     // std::vector

namespace plssvm::detail::tracking {

gpu_nvidia_hardware_sampler::gpu_nvidia_hardware_sampler(const std::size_t device_id, const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval } {
    // make sure that nvmlInit is only called once for all instances
    if (instances_++ == 0) {
        PLSSVM_NVML_ERROR_CHECK(nvmlInit());
        // notify that initialization has been finished
        init_finished_ = true;
    } else {
        // wait until init has been finished!
        while (!init_finished_) { }
    }

    // track the NVML version
    std::string version(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE, '\0');
    PLSSVM_NVML_ERROR_CHECK(nvmlSystemGetNVMLVersion(version.data(), NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE));
    version = version.substr(0, version.find_first_of('\0'));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "nvml_version", version }));

    // initialize samples -> can't be done beforehand since the device handle can only be initialized after a call to nvmlInit
    device_ = nvml_device_handle{ device_id };
}

gpu_nvidia_hardware_sampler::~gpu_nvidia_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->has_sampling_started() && !this->has_sampling_stopped()) {
            this->stop_sampling();
        }

        // the last instance must shut down the NVML runtime
        // make sure that nvmlShutdown is only called once
        if (--instances_ == 0) {
            PLSSVM_NVML_ERROR_CHECK(nvmlShutdown());
            // reset init_finished flag
            init_finished_ = false;
        }
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        std::terminate();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }
}

std::string gpu_nvidia_hardware_sampler::device_identification() const {
    nvmlPciInfo_st pcie_info{};
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPciInfo_v3(device_.get_impl().device, &pcie_info));
    return fmt::format("gpu_nvidia_device_{}_{}", pcie_info.bus, pcie_info.device);
}

target_platform gpu_nvidia_hardware_sampler::sampling_target() const {
    return target_platform::gpu_nvidia;
}

std::string gpu_nvidia_hardware_sampler::generate_yaml_string(const std::chrono::steady_clock::time_point start_time_point) const {
    // check whether it's safe to generate the YAML entry
    if (this->is_sampling()) {
        throw hardware_sampling_exception{ "Can't create the final YAML entry if the hardware sampler is still running!" };
    }

    return fmt::format("\n"
                       "    sampling_interval: {}\n"
                       "    time_points: [{}]\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}",
                       this->sampling_interval(),
                       fmt::join(durations_from_reference_time(this->time_points(), start_time_point), ", "),
                       general_samples_.generate_yaml_string(),
                       clock_samples_.generate_yaml_string(),
                       power_samples_.generate_yaml_string(),
                       memory_samples_.generate_yaml_string(),
                       temperature_samples_.generate_yaml_string());
}

void gpu_nvidia_hardware_sampler::sampling_loop() {
    // get the nvml handle from the device
    nvmlDevice_t device = device_.get_impl().device;

    //
    // add samples where we only have to retrieve the value once
    //

    this->add_time_point(std::chrono::steady_clock::now());

    // retrieve initial general information
    {
        // fixed information -> only retrieved once
        std::string name(NVML_DEVICE_NAME_V2_BUFFER_SIZE, '\0');
        if (nvmlDeviceGetName(device, name.data(), name.size()) == NVML_SUCCESS) {
            general_samples_.name_ = name.substr(0, name.find_first_of('\0'));
        }

        nvmlEnableState_t mode{};
        if (nvmlDeviceGetPersistenceMode(device, &mode) == NVML_SUCCESS) {
            general_samples_.persistence_mode_ = mode == NVML_FEATURE_ENABLED;
        }

        decltype(general_samples_.num_cores_)::value_type num_cores{};
        if (nvmlDeviceGetNumGpuCores(device, &num_cores) == NVML_SUCCESS) {
            general_samples_.num_cores_ = num_cores;
        }

        // queried samples -> retrieved every iteration if available
        nvmlPstates_t pstate{};
        if (nvmlDeviceGetPerformanceState(device, &pstate) == NVML_SUCCESS) {
            general_samples_.performance_state_ = decltype(general_samples_.performance_state_)::value_type{ static_cast<decltype(general_samples_.performance_state_)::value_type::value_type>(pstate) };
        }

        nvmlUtilization_t util{};
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
            general_samples_.utilization_gpu_ = decltype(general_samples_.utilization_gpu_)::value_type{ util.gpu };
            general_samples_.utilization_mem_ = decltype(general_samples_.utilization_gpu_)::value_type{ util.memory };
        }
    }

    // retrieve initial clock related information
    {
        // fixed information -> only retrieved once
        decltype(clock_samples_.adaptive_clock_status_)::value_type adaptive_clock_status{};
        if (nvmlDeviceGetAdaptiveClockInfoStatus(device, &adaptive_clock_status) == NVML_SUCCESS) {
            clock_samples_.adaptive_clock_status_ = adaptive_clock_status;
        }

        decltype(clock_samples_.clock_graph_max_)::value_type clock_graph_max{};
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &clock_graph_max) == NVML_SUCCESS) {
            clock_samples_.clock_graph_max_ = clock_graph_max;
        }

        decltype(clock_samples_.clock_sm_max_)::value_type clock_sm_max{};
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &clock_sm_max) == NVML_SUCCESS) {
            clock_samples_.clock_sm_max_ = clock_sm_max;
        }

        decltype(clock_samples_.clock_mem_max_)::value_type clock_mem_max{};
        if (nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock_mem_max) == NVML_SUCCESS) {
            clock_samples_.clock_mem_max_ = clock_mem_max;
        }

        {
            unsigned int clock_count{ 128 };
            std::vector<unsigned int> supported_clocks(clock_count);
            if (nvmlDeviceGetSupportedMemoryClocks(device, &clock_count, supported_clocks.data()) == NVML_SUCCESS) {
                supported_clocks.resize(clock_count);
                clock_samples_.clock_mem_min_ = *std::min_element(supported_clocks.cbegin(), supported_clocks.cend());
            }
        }

        {
            unsigned int clock_count{ 128 };
            std::vector<unsigned int> supported_clocks(clock_count);
            if (clock_samples_.clock_mem_min_.has_value() && nvmlDeviceGetSupportedGraphicsClocks(device, clock_samples_.clock_mem_min_.value(), &clock_count, supported_clocks.data()) == NVML_SUCCESS) {
                supported_clocks.resize(clock_count);
                clock_samples_.clock_graph_min_ = *std::min_element(supported_clocks.cbegin(), supported_clocks.cend());
            }
        }

        // queried samples -> retrieved every iteration if available
        decltype(clock_samples_.clock_graph_)::value_type::value_type clock_graph{};
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &clock_graph) == NVML_SUCCESS) {
            clock_samples_.clock_graph_ = decltype(clock_samples_.clock_graph_)::value_type{ clock_graph };
        }

        decltype(clock_samples_.clock_sm_)::value_type::value_type clock_sm{};
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock_sm) == NVML_SUCCESS) {
            clock_samples_.clock_sm_ = decltype(clock_samples_.clock_sm_)::value_type{ clock_sm };
        }

        decltype(clock_samples_.clock_mem_)::value_type::value_type clock_mem{};
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock_mem) == NVML_SUCCESS) {
            clock_samples_.clock_mem_ = decltype(clock_samples_.clock_mem_)::value_type{ clock_mem };
        }

        decltype(clock_samples_.clock_throttle_reason_)::value_type::value_type clock_throttle_reason{};
        if (nvmlDeviceGetCurrentClocksThrottleReasons(device, &clock_throttle_reason) == NVML_SUCCESS) {
            clock_samples_.clock_throttle_reason_ = decltype(clock_samples_.clock_throttle_reason_)::value_type{ clock_throttle_reason };
        }

        nvmlEnableState_t mode{};
        nvmlEnableState_t default_mode{};
        if (nvmlDeviceGetAutoBoostedClocksEnabled(device, &mode, &default_mode) == NVML_SUCCESS) {
            clock_samples_.auto_boosted_clocks_ = decltype(clock_samples_.auto_boosted_clocks_)::value_type{ mode == NVML_FEATURE_ENABLED };
        }
    }

    // retrieve initial power related information
    {
        // fixed information -> only retrieved once
        nvmlEnableState_t mode{};
        if (nvmlDeviceGetPowerManagementMode(device, &mode) == NVML_SUCCESS) {
            power_samples_.power_management_mode_ = mode == NVML_FEATURE_ENABLED;
        }

        decltype(power_samples_.power_management_limit_)::value_type power_management_limit{};
        if (nvmlDeviceGetPowerManagementLimit(device, &power_management_limit) == NVML_SUCCESS) {
            power_samples_.power_management_limit_ = power_management_limit;
        }

        decltype(power_samples_.power_enforced_limit_)::value_type power_enforced_limit{};
        if (nvmlDeviceGetEnforcedPowerLimit(device, &power_enforced_limit) == NVML_SUCCESS) {
            power_samples_.power_enforced_limit_ = power_enforced_limit;
        }

        // queried samples -> retrieved every iteration if available
        nvmlPstates_t pstate{};
        if (nvmlDeviceGetPowerState(device, &pstate) == NVML_SUCCESS) {
            power_samples_.power_state_ = decltype(general_samples_.performance_state_)::value_type{ static_cast<decltype(power_samples_.power_state_)::value_type::value_type>(pstate) };
        }

        decltype(power_samples_.power_usage_)::value_type::value_type power_usage{};
        if (nvmlDeviceGetPowerUsage(device, &power_usage) == NVML_SUCCESS) {
            power_samples_.power_usage_ = decltype(power_samples_.power_usage_)::value_type{ power_usage };
        }

        decltype(power_samples_.power_total_energy_consumption_)::value_type::value_type power_total_energy_consumption{};
        if (nvmlDeviceGetTotalEnergyConsumption(device, &power_total_energy_consumption) == NVML_SUCCESS) {
            power_samples_.power_total_energy_consumption_ = decltype(power_samples_.power_total_energy_consumption_)::value_type{ power_total_energy_consumption };
        }
    }

    // retrieve initial memory related information
    {
        // fixed information -> only retrieved once
        nvmlMemory_t memory_info{};
        if (nvmlDeviceGetMemoryInfo(device, &memory_info) == NVML_SUCCESS) {
            memory_samples_.memory_total_ = memory_info.total;
            // queried samples -> retrieved every iteration if available
            memory_samples_.memory_free_ = decltype(memory_samples_.memory_free_)::value_type{ memory_info.free };
            memory_samples_.memory_used_ = decltype(memory_samples_.memory_used_)::value_type{ memory_info.used };
        }

        decltype(memory_samples_.memory_bus_width_)::value_type memory_bus_width{};
        if (nvmlDeviceGetMemoryBusWidth(device, &memory_bus_width) == NVML_SUCCESS) {
            memory_samples_.memory_bus_width_ = memory_bus_width;
        }

        decltype(memory_samples_.max_pcie_link_generation_)::value_type max_pcie_link_generation{};
        if (nvmlDeviceGetMaxPcieLinkGeneration(device, &max_pcie_link_generation) == NVML_SUCCESS) {
            memory_samples_.max_pcie_link_generation_ = max_pcie_link_generation;
        }

        decltype(memory_samples_.pcie_link_max_speed_)::value_type pcie_link_max_speed{};
        if (nvmlDeviceGetPcieLinkMaxSpeed(device, &pcie_link_max_speed) == NVML_SUCCESS) {
            memory_samples_.pcie_link_max_speed_ = pcie_link_max_speed;
        }

        // queried samples -> retrieved every iteration if available
        decltype(memory_samples_.pcie_link_width_)::value_type::value_type pcie_link_width{};
        if (nvmlDeviceGetCurrPcieLinkWidth(device, &pcie_link_width) == NVML_SUCCESS) {
            memory_samples_.pcie_link_width_ = decltype(memory_samples_.pcie_link_width_)::value_type{ pcie_link_width };
        }

        decltype(memory_samples_.pcie_link_generation_)::value_type::value_type pcie_link_generation{};
        if (nvmlDeviceGetCurrPcieLinkGeneration(device, &pcie_link_generation) == NVML_SUCCESS) {
            memory_samples_.pcie_link_generation_ = decltype(memory_samples_.pcie_link_generation_)::value_type{ pcie_link_generation };
        }
    }

    // retrieve initial temperature related information
    {
        // fixed information -> only retrieved once
        decltype(temperature_samples_.num_fans_)::value_type num_fans{};
        if (nvmlDeviceGetNumFans(device, &num_fans) == NVML_SUCCESS) {
            temperature_samples_.num_fans_ = num_fans;
        }

        if (temperature_samples_.num_fans_.has_value() && temperature_samples_.num_fans_.value() > 0) {
            decltype(temperature_samples_.min_fan_speed_)::value_type min_fan_speed{};
            decltype(temperature_samples_.max_fan_speed_)::value_type max_fan_speed{};
            if (nvmlDeviceGetMinMaxFanSpeed(device, &min_fan_speed, &max_fan_speed) == NVML_SUCCESS) {
                temperature_samples_.min_fan_speed_ = min_fan_speed;
                temperature_samples_.max_fan_speed_ = max_fan_speed;
            }
        }

        decltype(temperature_samples_.temperature_threshold_gpu_max_)::value_type temperature_threshold_gpu_max{};
        if (nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &temperature_threshold_gpu_max) == NVML_SUCCESS) {
            temperature_samples_.temperature_threshold_gpu_max_ = temperature_threshold_gpu_max;
        }

        decltype(temperature_samples_.temperature_threshold_mem_max_)::value_type temperature_threshold_mem_max{};
        if (nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_MEM_MAX, &temperature_threshold_mem_max) == NVML_SUCCESS) {
            temperature_samples_.temperature_threshold_mem_max_ = temperature_threshold_mem_max;
        }

        // queried samples -> retrieved every iteration if available
        decltype(temperature_samples_.fan_speed_)::value_type::value_type fan_speed{};
        if (nvmlDeviceGetFanSpeed(device, &fan_speed) == NVML_SUCCESS) {
            temperature_samples_.fan_speed_ = decltype(temperature_samples_.fan_speed_)::value_type{ fan_speed };
        }

        decltype(temperature_samples_.temperature_gpu_)::value_type::value_type temperature_gpu{};
        if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature_gpu) == NVML_SUCCESS) {
            temperature_samples_.temperature_gpu_ = decltype(temperature_samples_.temperature_gpu_)::value_type{ temperature_gpu };
        }
    }

    //
    // loop until stop_sampling() is called
    //

    while (!this->has_sampling_stopped()) {
        // only sample values if the sampler currently isn't paused
        if (this->is_sampling()) {
            // add current time point
            this->add_time_point(std::chrono::steady_clock::now());

            // retrieve general samples
            {
                if (general_samples_.performance_state_.has_value()) {
                    nvmlPstates_t pstate{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPerformanceState(device, &pstate));
                    general_samples_.performance_state_->push_back(static_cast<decltype(general_samples_.performance_state_)::value_type::value_type>(pstate));
                }

                if (general_samples_.utilization_gpu_.has_value() && general_samples_.utilization_mem_.has_value()) {
                    nvmlUtilization_t util{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetUtilizationRates(device, &util));
                    general_samples_.utilization_gpu_->push_back(util.gpu);
                    general_samples_.utilization_mem_->push_back(util.memory);
                }
            }

            // retrieve clock related samples
            {
                if (clock_samples_.clock_graph_.has_value()) {
                    decltype(clock_samples_.clock_graph_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &value));
                    clock_samples_.clock_graph_->push_back(value);
                }

                if (clock_samples_.clock_sm_.has_value()) {
                    decltype(clock_samples_.clock_sm_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &value));
                    clock_samples_.clock_sm_->push_back(value);
                }

                if (clock_samples_.clock_mem_.has_value()) {
                    decltype(clock_samples_.clock_mem_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &value));
                    clock_samples_.clock_mem_->push_back(value);
                }

                if (clock_samples_.clock_throttle_reason_.has_value()) {
                    decltype(clock_samples_.clock_throttle_reason_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrentClocksThrottleReasons(device, &value));
                    clock_samples_.clock_throttle_reason_->push_back(value);
                }

                if (clock_samples_.auto_boosted_clocks_.has_value()) {
                    nvmlEnableState_t mode{};
                    nvmlEnableState_t default_mode{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetAutoBoostedClocksEnabled(device, &mode, &default_mode));
                    clock_samples_.auto_boosted_clocks_->push_back(mode == NVML_FEATURE_ENABLED);
                }
            }

            // retrieve power related information
            {
                if (power_samples_.power_state_.has_value()) {
                    nvmlPstates_t pstate{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerState(device, &pstate));
                    power_samples_.power_state_->push_back(static_cast<decltype(power_samples_.power_state_)::value_type::value_type>(pstate));
                }

                if (power_samples_.power_usage_.has_value()) {
                    decltype(power_samples_.power_usage_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerUsage(device, &value));
                    power_samples_.power_usage_->push_back(value);
                }

                if (power_samples_.power_total_energy_consumption_.has_value()) {
                    decltype(power_samples_.power_total_energy_consumption_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &value));
                    power_samples_.power_total_energy_consumption_->push_back(value);
                }
            }

            // retrieve memory related information
            {
                if (memory_samples_.memory_free_.has_value() && memory_samples_.memory_used_.has_value()) {
                    nvmlMemory_t memory_info{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMemoryInfo(device, &memory_info));
                    memory_samples_.memory_free_->push_back(memory_info.free);
                    memory_samples_.memory_used_->push_back(memory_info.used);
                }

                if (memory_samples_.pcie_link_width_.has_value()) {
                    decltype(memory_samples_.pcie_link_width_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrPcieLinkWidth(device, &value));
                    memory_samples_.pcie_link_width_->push_back(value);
                }

                if (memory_samples_.pcie_link_generation_.has_value()) {
                    decltype(memory_samples_.pcie_link_generation_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(device, &value));
                    memory_samples_.pcie_link_generation_->push_back(value);
                }
            }

            // retrieve temperature related information
            {
                if (temperature_samples_.fan_speed_.has_value()) {
                    decltype(temperature_samples_.fan_speed_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetFanSpeed(device, &value));
                    temperature_samples_.fan_speed_->push_back(value);
                }

                if (temperature_samples_.temperature_gpu_.has_value()) {
                    decltype(temperature_samples_.temperature_gpu_)::value_type::value_type value{};
                    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value));
                    temperature_samples_.temperature_gpu_->push_back(value);
                }
            }
        }

        // wait for the sampling interval to pass to retrieve the next sample
        std::this_thread::sleep_for(this->sampling_interval());
    }
}

std::ostream &operator<<(std::ostream &out, const gpu_nvidia_hardware_sampler &sampler) {
    if (sampler.is_sampling()) {
        out.setstate(std::ios_base::failbit);
        return out;
    } else {
        return out << fmt::format("sampling interval: {}\n"
                                  "time points: [{}]\n\n"
                                  "general samples:\n{}\n\n"
                                  "clock samples:\n{}\n\n"
                                  "power samples:\n{}\n\n"
                                  "memory samples:\n{}\n\n"
                                  "temperature samples:\n{}",
                                  sampler.sampling_interval(),
                                  fmt::join(time_points_to_epoch(sampler.time_points()), ", "),
                                  sampler.general_samples(),
                                  sampler.clock_samples(),
                                  sampler.power_samples(),
                                  sampler.memory_samples(),
                                  sampler.temperature_samples());
    }
}

}  // namespace plssvm::detail::tracking

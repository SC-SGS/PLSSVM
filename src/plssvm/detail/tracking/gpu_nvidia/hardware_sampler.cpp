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
#include "plssvm/detail/tracking/utility.hpp"                             // plssvm::detail::tracking::durations_from_reference_time
#include "plssvm/exceptions/exceptions.hpp"                               // plssvm::exception, plssvm::hardware_sampling_exception

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join
#include "nvml.h"        // NVML runtime functions

#include <algorithm>  // std::min_element
#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <exception>  // std::exception, std::terminate
#include <iostream>   // std::cerr, std::endl
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
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "nvml_version", version }));

    // initialize samples -> can't be done beforehand since the device handle can only be initialized after a call to nvmlInit
    device_ = nvml_device_handle{ device_id };
    general_samples_ = nvml_general_samples{ device_ };
    clock_samples_ = nvml_clock_samples{ device_ };
    power_samples_ = nvml_power_samples{ device_ };
    memory_samples_ = nvml_memory_samples{ device_ };
    temperature_samples_ = nvml_temperature_samples{ device_ };
}

gpu_nvidia_hardware_sampler::~gpu_nvidia_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->is_sampling()) {
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

std::string gpu_nvidia_hardware_sampler::generate_yaml_string(const std::chrono::system_clock::time_point start_time_point) const {
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
    // get the nvml handle from the device_id
    nvmlDevice_t device = device_.get_impl().device;
    //
    // add samples where we only have to retrieve the value once
    //

    // retrieve fixed general information
    {
        std::string name(NVML_DEVICE_NAME_V2_BUFFER_SIZE, '\0');
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetName(device, name.data(), name.size()));
        general_samples_.name = name;  //.substr(0, name.find_first_of('\0'));
        nvmlEnableState_t mode{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPersistenceMode(device, &mode));
        general_samples_.persistence_mode = mode == NVML_FEATURE_ENABLED;
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetNumGpuCores(device, &general_samples_.num_cores));
    }
    // retrieve fixed clock related information
    {
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetAdaptiveClockInfoStatus(device, &clock_samples_.adaptive_clock_status));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &clock_samples_.clock_graph_max));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &clock_samples_.clock_sm_max));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock_samples_.clock_mem_max));

        unsigned int clock_count{ 128 };
        std::vector<unsigned int> supported_clocks(clock_count);
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetSupportedMemoryClocks(device, &clock_count, supported_clocks.data()));
        supported_clocks.resize(clock_count);
        clock_samples_.clock_mem_min = *std::min_element(supported_clocks.cbegin(), supported_clocks.cend());

        clock_count = 128u;
        supported_clocks.resize(128);
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetSupportedGraphicsClocks(device, clock_samples_.clock_mem_min, &clock_count, supported_clocks.data()));
        clock_samples_.clock_graph_min = *std::min_element(supported_clocks.cbegin(), supported_clocks.cend());
    }
    // retrieve fixed power related information
    {
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerManagementLimit(device, &power_samples_.power_management_limit));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetEnforcedPowerLimit(device, &power_samples_.power_enforced_limit));
    }
    // retrieve fixed memory related information
    {
        nvmlMemory_t memory_info{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMemoryInfo(device, &memory_info));
        memory_samples_.memory_total = memory_info.total;
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMemoryBusWidth(device, &memory_samples_.memory_bus_width));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &memory_samples_.max_pcie_link_generation));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPcieLinkMaxSpeed(device, &memory_samples_.pcie_link_max_speed));
    }
    // retrieve fixed temperature related information
    {
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetNumFans(device, &temperature_samples_.num_fans));
        if (temperature_samples_.num_fans > 0) {
            PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMinMaxFanSpeed(device, &temperature_samples_.min_fan_speed, &temperature_samples_.max_fan_speed));
        }
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &temperature_samples_.temperature_threshold_gpu_max));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_MEM_MAX, &temperature_samples_.temperature_threshold_mem_max));
    }

    //
    // loop until stop_sampling() is called
    //

    while (!this->has_sampling_stopped()) {
        // only sample values if the sampler currently isn't paused
        if (this->is_sampling()) {
            // add current time point
            this->add_time_point(std::chrono::system_clock::now());

            // retrieve general information
            {
                nvml_general_samples::nvml_general_sample sample{};
                nvmlPstates_t pstate{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPerformanceState(device, &pstate));
                sample.performance_state = static_cast<int>(pstate);
                nvmlUtilization_t util{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetUtilizationRates(device, &util));
                sample.utilization_gpu = util.gpu;
                sample.utilization_mem = util.memory;
                general_samples_.add_sample(sample);
            }
            // retrieve clock related information
            {
                nvml_clock_samples::nvml_clock_sample sample{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &sample.clock_graph));
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sample.clock_sm));
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &sample.clock_mem));
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrentClocksThrottleReasons(device, &sample.clock_throttle_reason));
                nvmlEnableState_t mode{};
                nvmlEnableState_t default_mode{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetAutoBoostedClocksEnabled(device, &mode, &default_mode));
                sample.auto_boosted_clocks = mode == NVML_FEATURE_ENABLED;
                clock_samples_.add_sample(sample);
            }
            // retrieve power related information
            {
                nvml_power_samples::nvml_power_sample sample{};
                nvmlPstates_t pstate{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerState(device, &pstate));
                sample.power_state = static_cast<int>(pstate);
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerUsage(device, &sample.power_usage));
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &sample.power_total_energy_consumption));
                power_samples_.add_sample(sample);
            }
            // retrieve memory related information
            {
                nvml_memory_samples::nvml_memory_sample sample{};
                nvmlMemory_t memory_info{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMemoryInfo(device, &memory_info));
                sample.memory_free = memory_info.free;
                sample.memory_used = memory_info.used;
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrPcieLinkWidth(device, &sample.pcie_link_width));
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrPcieLinkGeneration(device, &sample.pcie_link_generation));
                memory_samples_.add_sample(sample);
            }
            // retrieve temperature related information
            {
                nvml_temperature_samples::nvml_temperature_sample sample{};
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetFanSpeed(device, &sample.fan_speed));
                PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &sample.temperature_gpu));
                temperature_samples_.add_sample(sample);
            }
        }

        // wait for the sampling interval to pass to retrieve the next sample
        std::this_thread::sleep_for(this->sampling_interval());
    }
}

}  // namespace plssvm::detail::tracking

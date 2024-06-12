/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/rocm_smi_hardware_sampler.hpp"

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/rocm_smi_samples.hpp"  // plssvm::detail::tracking::{rocm_smi_general_samples, rocm_smi_clock_samples, rocm_smi_power_samples, rocm_smi_memory_samples, rocm_smi_temperature_samples}
#include "plssvm/exceptions/exceptions.hpp"             // plssvm::exception, plssvm::hardware_sampling_exception

#include "fmt/chrono.h"         // format std::chrono types
#include "fmt/core.h"           // fmt::format
#include "fmt/format.h"         // fmt::join
#include "rocm_smi/rocm_smi.h"  // ROCm SMI runtime functions

#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <cstdint>    // std::uint32_t, std::uint64_t
#include <exception>  // std::exception, std::terminate
#include <iostream>   // std::cerr, std::endl
#include <string>     // std::string
#include <thread>     // std::this_thread
#include <vector>     // std::vector

namespace plssvm::detail::tracking {

#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)

    #define PLSSVM_ROCM_SMI_ERROR_CHECK(rocm_smi_func)                                                                     \
        {                                                                                                                  \
            const rsmi_status_t errc = rocm_smi_func;                                                                      \
            if (errc != RSMI_STATUS_SUCCESS && errc != RSMI_STATUS_NOT_SUPPORTED) {                                        \
                const char *error_string;                                                                                  \
                const rsmi_status_t ret = rsmi_status_string(errc, &error_string);                                         \
                if (ret == RSMI_STATUS_SUCCESS) {                                                                          \
                    throw hardware_sampling_exception{ fmt::format("Error in ROCm SMI function call: {}", error_string) }; \
                } else {                                                                                                   \
                    throw hardware_sampling_exception{ "Error in ROCm SMI function call" };                                \
                }                                                                                                          \
            }                                                                                                              \
        }

#else
    #define PLSSVM_ROCM_SMI_ERROR_CHECK(rocm_smi_func) rocm_smi_func;
#endif

rocm_smi_hardware_sampler::rocm_smi_hardware_sampler(const std::size_t device_id, const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval },
    device_id_{ static_cast<std::uint32_t>(device_id) },
    general_samples_{ static_cast<std::uint32_t>(device_id) },
    clock_samples_{ static_cast<std::uint32_t>(device_id) },
    power_samples_{ static_cast<std::uint32_t>(device_id) },
    memory_samples_{ static_cast<std::uint32_t>(device_id) },
    temperature_samples_{ static_cast<std::uint32_t>(device_id) } {
    // make sure that rsmi_init is only called once for all instances
    if (instances_++ == 0) {
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_init(std::uint64_t{ 0 }));
        // notify that initialization has been finished
        init_finished_ = true;
    } else {
        // wait until init has been finished!
        while (!init_finished_) { }
    }
}

rocm_smi_hardware_sampler::~rocm_smi_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->is_sampling()) {
            this->stop_sampling();
        }

        // the last instance must shut down the ROCm SMI runtime
        // make sure that rsmi_shut_down is only called once
        if (--instances_ == 0) {
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_shut_down());
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

std::string rocm_smi_hardware_sampler::device_identification() const noexcept {
    return fmt::format("rocm_smi_device_{}", device_id_);
}

std::string rocm_smi_hardware_sampler::assemble_yaml_sample_string() const {
    // check whether it's safe to generate the YAML entry
    if (this->is_sampling()) {
        throw hardware_sampling_exception{ "Can't create the final YAML entry if the hardware sampler is still running!" };
    }

    return fmt::format("\n"
                       "    samples:\n"
                       "      sampling_interval: {}\n"
                       "      time_points: [{}]\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}",
                       this->sampling_interval(),
                       fmt::join(time_since_start_, ", "),
                       general_samples_,
                       clock_samples_,
                       power_samples_,
                       memory_samples_,
                       temperature_samples_);
}

void rocm_smi_hardware_sampler::sampling_loop() {
    //
    // add samples where we only have to retrieve the value once
    //

    // retrieve fixed general information
    {
        std::string name(static_cast<std::string::size_type>(1024), '\0');
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_name_get(device_id_, name.data(), name.size()));
        general_samples_.name = name;
    }
    // retrieve fixed clock related information
    {
        rsmi_frequencies_t frequency_info{};
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SYS, &frequency_info));
        clock_samples_.clock_system_min = frequency_info.frequency[0];
        clock_samples_.clock_system_max = frequency_info.frequency[frequency_info.num_supported - 1];
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SOC, &frequency_info));
        clock_samples_.clock_socket_min = frequency_info.frequency[0];
        clock_samples_.clock_socket_max = frequency_info.frequency[frequency_info.num_supported - 1];
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_MEM, &frequency_info));
        clock_samples_.clock_memory_min = frequency_info.frequency[0];
        clock_samples_.clock_memory_max = frequency_info.frequency[frequency_info.num_supported - 1];
    }
    // retrieve fixed power related information
    {
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_cap_default_get(device_id_, &power_samples_.power_default_cap));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_cap_get(device_id_, std::uint32_t{ 0 }, &power_samples_.power_cap));
        [[maybe_unused]] std::uint64_t power_usage{};
        RSMI_POWER_TYPE power_type{};
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_get(device_id_, &power_usage, &power_type));
        switch (power_type) {
            case RSMI_POWER_TYPE::RSMI_AVERAGE_POWER:
                power_samples_.power_type = "average";
                break;
            case RSMI_POWER_TYPE::RSMI_CURRENT_POWER:
                power_samples_.power_type = "current/instant";
                break;
            case RSMI_POWER_TYPE::RSMI_INVALID_POWER:
                power_samples_.power_type = "invalid/undetected";
                break;
        }
        // rsmi_power_profile_status_t power_profile{};
        // PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_profile_presets_get(device_id_, std::uint32_t{ 0 }, &power_profile));
        // power_samples_.available_power_profiles = static_cast<std::uint64_t>(power_profile.available_profiles);
    }
    // retrieve fixed memory related information
    {
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_memory_total_get(device_id_, RSMI_MEM_TYPE_VRAM, &memory_samples_.memory_total));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_memory_total_get(device_id_, RSMI_MEM_TYPE_VIS_VRAM, &memory_samples_.visible_memory_total));
        rsmi_pcie_bandwidth_t bandwidth_info{};
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_pci_bandwidth_get(device_id_, &bandwidth_info));
        memory_samples_.min_num_pcie_lanes = bandwidth_info.lanes[0];
        memory_samples_.max_num_pcie_lanes = bandwidth_info.lanes[bandwidth_info.transfer_rate.num_supported - 1];
    }
    // retrieve fixed temperature related information
    {
        std::uint32_t fan_id{ 0 };
        [[maybe_unused]] std::int64_t fan_speed{ 0 };
        while (rsmi_dev_fan_speed_get(device_id_, fan_id, &fan_speed) == RSMI_STATUS_SUCCESS) {
            ++fan_id;
        }
        temperature_samples_.num_fans = fan_id;

        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_fan_speed_max_get(device_id_, std::uint32_t{ 0 }, &temperature_samples_.max_fan_speed));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MIN, &temperature_samples_.temperature_edge_min));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MAX, &temperature_samples_.temperature_edge_max));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_MIN, &temperature_samples_.temperature_hotspot_min));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_MAX, &temperature_samples_.temperature_hotspot_max));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_MIN, &temperature_samples_.temperature_memory_min));
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_MAX, &temperature_samples_.temperature_memory_max));
    }

    //
    // loop until stop_sampling() is called
    //

    while (!sampling_stopped_) {
        // add current time point
        time_since_start_.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - this->sampling_steady_clock_start_time()));

        // retrieve general information
        {
            rocm_smi_general_samples::rocm_smi_general_sample sample{};
            rsmi_dev_perf_level_t pstate{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_perf_level_get(device_id_, &pstate));
            sample.performance_state = static_cast<int>(pstate);
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_busy_percent_get(device_id_, &sample.utilization_gpu));
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_memory_busy_percent_get(device_id_, &sample.utilization_mem));
            general_samples_.add_sample(sample);
        }
        // retrieve clock related information
        {
            rocm_smi_clock_samples::rocm_smi_clock_sample sample{};
            rsmi_frequencies_t frequency_info{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SYS, &frequency_info));
            sample.clock_system = frequency_info.frequency[frequency_info.current];
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SOC, &frequency_info));
            sample.clock_socket = frequency_info.frequency[frequency_info.current];
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_MEM, &frequency_info));
            sample.clock_memory = frequency_info.frequency[frequency_info.current];
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_metrics_throttle_status_get(device_id_, &sample.clock_throttle_reason));
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_overdrive_level_get(device_id_, &sample.overdrive_level));
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_mem_overdrive_level_get(device_id_, &sample.memory_overdrive_level));
            clock_samples_.add_sample(sample);
        }
        // retrieve power related information
        {
            rocm_smi_power_samples::rocm_smi_power_sample sample{};
            [[maybe_unused]] RSMI_POWER_TYPE power_type{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_get(device_id_, &sample.power_usage, &power_type));
            [[maybe_unused]] std::uint64_t timestamp{};
            [[maybe_unused]] float resolution{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_energy_count_get(device_id_, &sample.power_total_energy_consumption, &resolution, &timestamp));  // TODO: returns the same value for all invocations
            // rsmi_power_profile_status_t power_profile{};
            // PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_profile_presets_get(device_id_, std::uint32_t{ 0 }, &power_profile));
            // sample.power_profile = static_cast<std::uint64_t>(power_profile.current);
            power_samples_.add_sample(sample);
        }
        // retrieve memory related information
        {
            rocm_smi_memory_samples::rocm_smi_memory_sample sample{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_memory_usage_get(device_id_, RSMI_MEM_TYPE_VRAM, &sample.memory_used));
            rsmi_pcie_bandwidth_t bandwidth_info{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_pci_bandwidth_get(device_id_, &bandwidth_info));
            sample.pcie_transfer_rate = bandwidth_info.transfer_rate.frequency[bandwidth_info.transfer_rate.current];
            sample.num_pcie_lanes = bandwidth_info.lanes[bandwidth_info.transfer_rate.current];
            memory_samples_.add_sample(sample);
        }
        // retrieve temperature related information
        {
            rocm_smi_temperature_samples::rocm_smi_temperature_sample sample{};
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_fan_speed_get(device_id_, std::uint32_t{ 0 }, &sample.fan_speed));
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &sample.temperature_edge));
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &sample.temperature_hotspot));
            PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT, &sample.temperature_memory));
            temperature_samples_.add_sample(sample);
        }

        // wait for sampling_interval_ milliseconds to retrieve the next sample
        std::this_thread::sleep_for(std::chrono::milliseconds{ this->sampling_interval() });
    }
}

}  // namespace plssvm::detail::tracking

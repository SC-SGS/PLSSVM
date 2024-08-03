/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_amd/hardware_sampler.hpp"

#include "plssvm/detail/tracking/gpu_amd/rocm_smi_samples.hpp"  // plssvm::detail::tracking::{rocm_smi_general_samples, rocm_smi_clock_samples, rocm_smi_power_samples, rocm_smi_memory_samples, rocm_smi_temperature_samples}
#include "plssvm/detail/tracking/gpu_amd/utility.hpp"           // PLSSVM_ROCM_SMI_ERROR_CHECK
#include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/performance_tracker.hpp"       // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/tracking/utility.hpp"                   // plssvm::detail::tracking::{durations_from_reference_time, time_points_to_epoch}
#include "plssvm/exceptions/exceptions.hpp"                     // plssvm::exception, plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"                          // plssvm::target_platform

#include "fmt/chrono.h"         // format std::chrono types
#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "rocm_smi/rocm_smi.h"  // ROCm SMI runtime functions

#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <cstdint>    // std::uint32_t, std::uint64_t
#include <exception>  // std::exception, std::terminate
#include <ios>        // std::ios_base
#include <iostream>   // std::cerr, std::endl
#include <optional>   // std::optional
#include <ostream>    // std::ostream
#include <string>     // std::string
#include <thread>     // std::this_thread
#include <utility>    // std::move
#include <vector>     // std::vector

namespace plssvm::detail::tracking {

gpu_amd_hardware_sampler::gpu_amd_hardware_sampler(const std::size_t device_id, const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval },
    device_id_{ static_cast<std::uint32_t>(device_id) } {
    // make sure that rsmi_init is only called once for all instances
    if (instances_++ == 0) {
        PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_init(std::uint64_t{ 0 }));
        // notify that initialization has been finished
        init_finished_ = true;
    } else {
        // wait until init has been finished!
        while (!init_finished_) { }
    }

    // track the ROCm SMI version
    rsmi_version_t version{};
    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_version_get(&version));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "rocm_smi_version", fmt::format("{}.{}.{}_{}", version.major, version.minor, version.patch, version.build) }));
}

gpu_amd_hardware_sampler::~gpu_amd_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->has_sampling_started() && !this->has_sampling_stopped()) {
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

std::string gpu_amd_hardware_sampler::device_identification() const {
    return fmt::format("gpu_amd_device_{}", device_id_);
}

target_platform gpu_amd_hardware_sampler::sampling_target() const {
    return target_platform::gpu_amd;
}

std::string gpu_amd_hardware_sampler::generate_yaml_string(const std::chrono::steady_clock::time_point start_time_point) const {
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

void gpu_amd_hardware_sampler::sampling_loop() {
    //
    // add samples where we only have to retrieve the value once
    //

    this->add_time_point(std::chrono::steady_clock::now());

    // retrieve initial general information
    {
        // fixed information -> only retrieved once
        std::string name(static_cast<std::string::size_type>(1024), '\0');
        if (rsmi_dev_name_get(device_id_, name.data(), name.size()) == RSMI_STATUS_SUCCESS) {
            general_samples_.name_ = name.substr(0, name.find_first_of('\0'));
        }

        // queried samples -> retrieved every iteration if available
        rsmi_dev_perf_level_t pstate{};
        if (rsmi_dev_perf_level_get(device_id_, &pstate) == RSMI_STATUS_SUCCESS) {
            general_samples_.performance_level_ = decltype(general_samples_.performance_level_)::value_type{ static_cast<decltype(general_samples_.performance_level_)::value_type::value_type>(pstate) };
        }

        decltype(general_samples_.utilization_gpu_)::value_type::value_type utilization_gpu{};
        if (rsmi_dev_busy_percent_get(device_id_, &utilization_gpu) == RSMI_STATUS_SUCCESS) {
            general_samples_.utilization_gpu_ = decltype(general_samples_.utilization_gpu_)::value_type{ utilization_gpu };
        }

        decltype(general_samples_.utilization_mem_)::value_type::value_type utilization_mem{};
        if (rsmi_dev_memory_busy_percent_get(device_id_, &utilization_mem) == RSMI_STATUS_SUCCESS) {
            general_samples_.utilization_mem_ = decltype(general_samples_.utilization_mem_)::value_type{ utilization_mem };
        }
    }

    // retrieve initial clock related information
    {
        rsmi_frequencies_t frequency_info{};
        if (rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SYS, &frequency_info) == RSMI_STATUS_SUCCESS) {
            clock_samples_.clock_system_min_ = frequency_info.frequency[0];
            clock_samples_.clock_system_max_ = frequency_info.frequency[frequency_info.num_supported - 1];
            // queried samples -> retrieved every iteration if available
            clock_samples_.clock_system_ = decltype(clock_samples_.clock_system_)::value_type{};
            if (frequency_info.current < RSMI_MAX_NUM_FREQUENCIES) {
                clock_samples_.clock_system_->push_back(frequency_info.frequency[frequency_info.current]);
            } else {
                clock_samples_.clock_system_->push_back(0);
            }
        }

        if (rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SOC, &frequency_info) == RSMI_STATUS_SUCCESS) {
            clock_samples_.clock_socket_min_ = frequency_info.frequency[0];
            clock_samples_.clock_socket_max_ = frequency_info.frequency[frequency_info.num_supported - 1];
            // queried samples -> retrieved every iteration if available
            clock_samples_.clock_socket_ = decltype(clock_samples_.clock_socket_)::value_type{};
            if (frequency_info.current < RSMI_MAX_NUM_FREQUENCIES) {
                clock_samples_.clock_socket_->push_back(frequency_info.frequency[frequency_info.current]);
            } else {
                clock_samples_.clock_socket_->push_back(0);
            }
        }

        if (rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_MEM, &frequency_info) == RSMI_STATUS_SUCCESS) {
            clock_samples_.clock_memory_min_ = frequency_info.frequency[0];
            clock_samples_.clock_memory_max_ = frequency_info.frequency[frequency_info.num_supported - 1];
            // queried samples -> retrieved every iteration if available
            clock_samples_.clock_memory_ = decltype(clock_samples_.clock_memory_)::value_type{};
            if (frequency_info.current < RSMI_MAX_NUM_FREQUENCIES) {
                clock_samples_.clock_memory_->push_back(frequency_info.frequency[frequency_info.current]);
            } else {
                clock_samples_.clock_memory_->push_back(0);
            }
        }

        // queried samples -> retrieved every iteration if available
        decltype(clock_samples_.overdrive_level_)::value_type::value_type overdrive_level{};
        if (rsmi_dev_overdrive_level_get(device_id_, &overdrive_level) == RSMI_STATUS_SUCCESS) {
            clock_samples_.overdrive_level_ = decltype(clock_samples_.overdrive_level_)::value_type{ overdrive_level };
        }

        decltype(clock_samples_.memory_overdrive_level_)::value_type::value_type memory_overdrive_level{};
        if (rsmi_dev_mem_overdrive_level_get(device_id_, &memory_overdrive_level) == RSMI_STATUS_SUCCESS) {
            clock_samples_.memory_overdrive_level_ = decltype(clock_samples_.memory_overdrive_level_)::value_type{ memory_overdrive_level };
        }
    }

    // retrieve initial power related information
    {
        decltype(power_samples_.power_default_cap_)::value_type power_default_cap{};
        if (rsmi_dev_power_cap_default_get(device_id_, &power_default_cap) == RSMI_STATUS_SUCCESS) {
            power_samples_.power_default_cap_ = power_default_cap;
        }

        decltype(power_samples_.power_cap_)::value_type power_cap{};
        if (rsmi_dev_power_cap_get(device_id_, std::uint32_t{ 0 }, &power_cap) == RSMI_STATUS_SUCCESS) {
            power_samples_.power_cap_ = power_cap;
        }

        {
            decltype(power_samples_.power_usage_)::value_type::value_type power_usage{};
            RSMI_POWER_TYPE power_type{};
            if (rsmi_dev_power_get(device_id_, &power_usage, &power_type) == RSMI_STATUS_SUCCESS) {
                switch (power_type) {
                    case RSMI_POWER_TYPE::RSMI_AVERAGE_POWER:
                        power_samples_.power_type_ = "average";
                        break;
                    case RSMI_POWER_TYPE::RSMI_CURRENT_POWER:
                        power_samples_.power_type_ = "current/instant";
                        break;
                    case RSMI_POWER_TYPE::RSMI_INVALID_POWER:
                        power_samples_.power_type_ = "invalid/undetected";
                        break;
                }
                // queried samples -> retrieved every iteration if available
                power_samples_.power_usage_ = decltype(power_samples_.power_usage_)::value_type{ power_usage };
            }
        }

        rsmi_power_profile_status_t power_profile{};
        if (rsmi_dev_power_profile_presets_get(device_id_, std::uint32_t{ 0 }, &power_profile) == RSMI_STATUS_SUCCESS) {
            decltype(power_samples_.available_power_profiles_)::value_type available_power_profiles{};
            // go through all possible power profiles
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_CUSTOM_MASK) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("CUSTOM");
            }
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_VIDEO_MASK) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("VIDEO");
            }
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_POWER_SAVING_MASK) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("POWER_SAVING");
            }
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_COMPUTE_MASK) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("COMPUTE");
            }
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_VR_MASK) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("VR");
            }
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("3D_FULL_SCREEN");
            }
            if ((power_profile.available_profiles & RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT) != std::uint64_t{ 0 }) {
                available_power_profiles.emplace_back("BOOTUP_DEFAULT");
            }
            power_samples_.available_power_profiles_ = std::move(available_power_profiles);

            // queried samples -> retrieved every iteration if available
            switch (power_profile.current) {
                case RSMI_PWR_PROF_PRST_CUSTOM_MASK:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "CUSTOM" };
                    break;
                case RSMI_PWR_PROF_PRST_VIDEO_MASK:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "VIDEO" };
                    break;
                case RSMI_PWR_PROF_PRST_POWER_SAVING_MASK:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "POWER_SAVING" };
                    break;
                case RSMI_PWR_PROF_PRST_COMPUTE_MASK:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "COMPUTE" };
                    break;
                case RSMI_PWR_PROF_PRST_VR_MASK:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "VR" };
                    break;
                case RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "3D_FULL_SCREEN" };
                    break;
                case RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "BOOTUP_DEFAULT" };
                    break;
                case RSMI_PWR_PROF_PRST_INVALID:
                    power_samples_.power_profile_ = decltype(power_samples_.power_profile_)::value_type{ "INVALID" };
                    break;
            }
        }

        // queried samples -> retrieved every iteration if available
        [[maybe_unused]] std::uint64_t timestamp{};
        [[maybe_unused]] float resolution{};
        decltype(power_samples_.power_total_energy_consumption_)::value_type::value_type power_total_energy_consumption{};
        if (rsmi_dev_energy_count_get(device_id_, &power_total_energy_consumption, &resolution, &timestamp) == RSMI_STATUS_SUCCESS) {  // TODO: returns the same value for all invocations
            power_samples_.power_total_energy_consumption_ = decltype(power_samples_.power_total_energy_consumption_)::value_type{ power_total_energy_consumption };
        }
    }

    // retrieve initial memory related information
    {
        decltype(memory_samples_.memory_total_)::value_type memory_total{};
        if (rsmi_dev_memory_total_get(device_id_, RSMI_MEM_TYPE_VRAM, &memory_total) == RSMI_STATUS_SUCCESS) {
            memory_samples_.memory_total_ = memory_total;
        }

        decltype(memory_samples_.visible_memory_total_)::value_type visible_memory_total{};
        if (rsmi_dev_memory_total_get(device_id_, RSMI_MEM_TYPE_VIS_VRAM, &visible_memory_total) == RSMI_STATUS_SUCCESS) {
            memory_samples_.visible_memory_total_ = visible_memory_total;
        }

        rsmi_pcie_bandwidth_t bandwidth_info{};
        if (rsmi_dev_pci_bandwidth_get(device_id_, &bandwidth_info) == RSMI_STATUS_SUCCESS) {
            memory_samples_.min_num_pcie_lanes_ = bandwidth_info.lanes[0];
            memory_samples_.max_num_pcie_lanes_ = bandwidth_info.lanes[bandwidth_info.transfer_rate.num_supported - 1];
            // queried samples -> retrieved every iteration if available
            memory_samples_.pcie_transfer_rate_ = decltype(memory_samples_.pcie_transfer_rate_)::value_type{};
            memory_samples_.num_pcie_lanes_ = decltype(memory_samples_.num_pcie_lanes_)::value_type{};
            if (bandwidth_info.transfer_rate.current < RSMI_MAX_NUM_FREQUENCIES) {
                memory_samples_.pcie_transfer_rate_->push_back(bandwidth_info.transfer_rate.frequency[bandwidth_info.transfer_rate.current]);
                memory_samples_.num_pcie_lanes_->push_back(bandwidth_info.lanes[bandwidth_info.transfer_rate.current]);
            } else {
                // the current index is (somehow) wrong
                memory_samples_.pcie_transfer_rate_->push_back(0);
                memory_samples_.num_pcie_lanes_->push_back(0);
            }
        }

        // queried samples -> retrieved every iteration if available
        decltype(memory_samples_.memory_used_)::value_type::value_type memory_used{};
        if (rsmi_dev_memory_usage_get(device_id_, RSMI_MEM_TYPE_VRAM, &memory_used) == RSMI_STATUS_SUCCESS) {
            memory_samples_.memory_used_ = decltype(memory_samples_.memory_used_)::value_type{ memory_used };
        }
    }

    // retrieve fixed temperature related information
    {
        std::uint32_t fan_id{ 0 };
        decltype(temperature_samples_.fan_speed_)::value_type::value_type fan_speed{};
        while (rsmi_dev_fan_speed_get(device_id_, fan_id, &fan_speed) == RSMI_STATUS_SUCCESS) {
            if (fan_id == 0) {
                // queried samples -> retrieved every iteration if available
                temperature_samples_.fan_speed_ = decltype(temperature_samples_.fan_speed_)::value_type{ fan_speed };
            }
            ++fan_id;
        }
        temperature_samples_.num_fans_ = fan_id;

        decltype(temperature_samples_.max_fan_speed_)::value_type max_fan_speed{};
        if (rsmi_dev_fan_speed_max_get(device_id_, std::uint32_t{ 0 }, &max_fan_speed) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.max_fan_speed_ = max_fan_speed;
        }

        decltype(temperature_samples_.temperature_edge_min_)::value_type temperature_edge_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MIN, &temperature_edge_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_edge_min_ = temperature_edge_min;
        }

        decltype(temperature_samples_.temperature_edge_max_)::value_type temperature_edge_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MAX, &temperature_edge_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_edge_max_ = temperature_edge_min;
        }

        decltype(temperature_samples_.temperature_hotspot_min_)::value_type temperature_hotspot_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_MIN, &temperature_hotspot_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hotspot_min_ = temperature_hotspot_min;
        }

        decltype(temperature_samples_.temperature_hotspot_max_)::value_type temperature_hotspot_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_MAX, &temperature_hotspot_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hotspot_max_ = temperature_hotspot_max;
        }

        decltype(temperature_samples_.temperature_memory_min_)::value_type temperature_memory_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_MIN, &temperature_memory_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_memory_min_ = temperature_memory_min;
        }

        decltype(temperature_samples_.temperature_memory_max_)::value_type temperature_memory_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_MAX, &temperature_memory_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_memory_max_ = temperature_memory_max;
        }

        decltype(temperature_samples_.temperature_hbm_0_min_)::value_type temperature_hbm_0_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_0, RSMI_TEMP_MIN, &temperature_hbm_0_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_0_min_ = temperature_hbm_0_min;
        }

        decltype(temperature_samples_.temperature_hbm_0_max_)::value_type temperature_hbm_0_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_0, RSMI_TEMP_MAX, &temperature_hbm_0_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_0_max_ = temperature_hbm_0_max;
        }

        decltype(temperature_samples_.temperature_hbm_1_min_)::value_type temperature_hbm_1_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_1, RSMI_TEMP_MIN, &temperature_hbm_1_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_1_min_ = temperature_hbm_1_min;
        }

        decltype(temperature_samples_.temperature_hbm_1_max_)::value_type temperature_hbm_1_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_1, RSMI_TEMP_MAX, &temperature_hbm_1_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_1_max_ = temperature_hbm_1_max;
        }

        decltype(temperature_samples_.temperature_hbm_2_min_)::value_type temperature_hbm_2_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_2, RSMI_TEMP_MIN, &temperature_hbm_2_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_2_min_ = temperature_hbm_2_min;
        }

        decltype(temperature_samples_.temperature_hbm_2_max_)::value_type temperature_hbm_2_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_2, RSMI_TEMP_MAX, &temperature_hbm_2_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_2_max_ = temperature_hbm_2_max;
        }

        decltype(temperature_samples_.temperature_hbm_3_min_)::value_type temperature_hbm_3_min{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_3, RSMI_TEMP_MIN, &temperature_hbm_3_min) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_3_min_ = temperature_hbm_3_min;
        }

        decltype(temperature_samples_.temperature_hbm_3_max_)::value_type temperature_hbm_3_max{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_3, RSMI_TEMP_MAX, &temperature_hbm_3_max) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_3_max_ = temperature_hbm_3_max;
        }

        // queried samples -> retrieved every iteration if available
        decltype(temperature_samples_.temperature_edge_)::value_type::value_type temperature_edge{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &temperature_edge) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_edge_ = decltype(temperature_samples_.temperature_edge_)::value_type{ temperature_edge };
        }

        decltype(temperature_samples_.temperature_hotspot_)::value_type::value_type temperature_hotspot{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &temperature_hotspot) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hotspot_ = decltype(temperature_samples_.temperature_hotspot_)::value_type{ temperature_hotspot };
        }

        decltype(temperature_samples_.temperature_memory_)::value_type::value_type temperature_memory{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT, &temperature_memory) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_memory_ = decltype(temperature_samples_.temperature_memory_)::value_type{ temperature_memory };
        }

        decltype(temperature_samples_.temperature_hbm_0_)::value_type::value_type temperature_hbm_0{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_0, RSMI_TEMP_CURRENT, &temperature_hbm_0) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_0_ = decltype(temperature_samples_.temperature_hbm_0_)::value_type{ temperature_hbm_0 };
        }

        decltype(temperature_samples_.temperature_hbm_1_)::value_type::value_type temperature_hbm_1{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_1, RSMI_TEMP_CURRENT, &temperature_hbm_1) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_1_ = decltype(temperature_samples_.temperature_hbm_1_)::value_type{ temperature_hbm_1 };
        }

        decltype(temperature_samples_.temperature_hbm_2_)::value_type::value_type temperature_hbm_2{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_2, RSMI_TEMP_CURRENT, &temperature_hbm_2) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_2_ = decltype(temperature_samples_.temperature_hbm_2_)::value_type{ temperature_hbm_2 };
        }

        decltype(temperature_samples_.temperature_hbm_3_)::value_type::value_type temperature_hbm_3{};
        if (rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_3, RSMI_TEMP_CURRENT, &temperature_hbm_3) == RSMI_STATUS_SUCCESS) {
            temperature_samples_.temperature_hbm_3_ = decltype(temperature_samples_.temperature_hbm_3_)::value_type{ temperature_hbm_3 };
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
                if (general_samples_.performance_level_.has_value()) {
                    rsmi_dev_perf_level_t pstate{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_perf_level_get(device_id_, &pstate));
                    general_samples_.performance_level_->push_back(static_cast<decltype(general_samples_.performance_level_)::value_type::value_type>(pstate));
                }

                if (general_samples_.utilization_gpu_.has_value()) {
                    decltype(general_samples_.utilization_gpu_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_busy_percent_get(device_id_, &value));
                    general_samples_.utilization_gpu_->push_back(value);
                }

                if (general_samples_.utilization_mem_.has_value()) {
                    decltype(general_samples_.utilization_mem_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_memory_busy_percent_get(device_id_, &value));
                    general_samples_.utilization_mem_->push_back(value);
                }
            }

            // retrieve clock related samples
            {
                if (clock_samples_.clock_system_.has_value()) {
                    rsmi_frequencies_t frequency_info{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SYS, &frequency_info));
                    if (frequency_info.current < RSMI_MAX_NUM_FREQUENCIES) {
                        clock_samples_.clock_system_->push_back(frequency_info.frequency[frequency_info.current]);
                    } else {
                        // the current index is (somehow) wrong
                        clock_samples_.clock_system_->push_back(0);
                    }
                }

                if (clock_samples_.clock_socket_.has_value()) {
                    rsmi_frequencies_t frequency_info{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_SOC, &frequency_info));
                    if (frequency_info.current < RSMI_MAX_NUM_FREQUENCIES) {
                        clock_samples_.clock_socket_->push_back(frequency_info.frequency[frequency_info.current]);
                    } else {
                        // the current index is (somehow) wrong
                        clock_samples_.clock_socket_->push_back(0);
                    }
                }

                if (clock_samples_.clock_memory_.has_value()) {
                    rsmi_frequencies_t frequency_info{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_gpu_clk_freq_get(device_id_, RSMI_CLK_TYPE_MEM, &frequency_info));
                    if (frequency_info.current < RSMI_MAX_NUM_FREQUENCIES) {
                        clock_samples_.clock_memory_->push_back(frequency_info.frequency[frequency_info.current]);
                    } else {
                        // the current index is (somehow) wrong
                        clock_samples_.clock_memory_->push_back(0);
                    }
                }

                if (clock_samples_.overdrive_level_.has_value()) {
                    decltype(clock_samples_.overdrive_level_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_overdrive_level_get(device_id_, &value));
                    clock_samples_.overdrive_level_->push_back(value);
                }

                if (clock_samples_.memory_overdrive_level_.has_value()) {
                    decltype(clock_samples_.memory_overdrive_level_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_mem_overdrive_level_get(device_id_, &value));
                    clock_samples_.memory_overdrive_level_->push_back(value);
                }
            }

            // retrieve power related samples
            {
                if (power_samples_.power_usage_.has_value()) {
                    [[maybe_unused]] RSMI_POWER_TYPE power_type{};
                    decltype(power_samples_.power_usage_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_get(device_id_, &value, &power_type));
                    power_samples_.power_usage_->push_back(value);
                }

                if (power_samples_.power_total_energy_consumption_.has_value()) {
                    [[maybe_unused]] std::uint64_t timestamp{};
                    [[maybe_unused]] float resolution{};
                    decltype(power_samples_.power_total_energy_consumption_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_energy_count_get(device_id_, &value, &resolution, &timestamp));  // TODO: returns the same value for all invocations
                    power_samples_.power_total_energy_consumption_->push_back(value);
                }

                if (power_samples_.power_profile_.has_value()) {
                    rsmi_power_profile_status_t power_profile{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_power_profile_presets_get(device_id_, std::uint32_t{ 0 }, &power_profile));
                    switch (power_profile.current) {
                        case RSMI_PWR_PROF_PRST_CUSTOM_MASK:
                            power_samples_.power_profile_->emplace_back("CUSTOM");
                            break;
                        case RSMI_PWR_PROF_PRST_VIDEO_MASK:
                            power_samples_.power_profile_->emplace_back("VIDEO");
                            break;
                        case RSMI_PWR_PROF_PRST_POWER_SAVING_MASK:
                            power_samples_.power_profile_->emplace_back("POWER_SAVING");
                            break;
                        case RSMI_PWR_PROF_PRST_COMPUTE_MASK:
                            power_samples_.power_profile_->emplace_back("COMPUTE");
                            break;
                        case RSMI_PWR_PROF_PRST_VR_MASK:
                            power_samples_.power_profile_->emplace_back("VR");
                            break;
                        case RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK:
                            power_samples_.power_profile_->emplace_back("3D_FULL_SCREEN");
                            break;
                        case RSMI_PWR_PROF_PRST_BOOTUP_DEFAULT:
                            power_samples_.power_profile_->emplace_back("BOOTUP_DEFAULT");
                            break;
                        case RSMI_PWR_PROF_PRST_INVALID:
                            power_samples_.power_profile_->emplace_back("INVALID");
                            break;
                    }
                }
            }

            // retrieve memory related samples
            {
                if (memory_samples_.memory_used_.has_value()) {
                    decltype(memory_samples_.memory_used_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_memory_usage_get(device_id_, RSMI_MEM_TYPE_VRAM, &value));
                    memory_samples_.memory_used_->push_back(value);
                }

                if (memory_samples_.pcie_transfer_rate_.has_value() && memory_samples_.num_pcie_lanes_.has_value()) {
                    rsmi_pcie_bandwidth_t bandwidth_info{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_pci_bandwidth_get(device_id_, &bandwidth_info));
                    if (bandwidth_info.transfer_rate.current < RSMI_MAX_NUM_FREQUENCIES) {
                        memory_samples_.pcie_transfer_rate_->push_back(bandwidth_info.transfer_rate.frequency[bandwidth_info.transfer_rate.current]);
                        memory_samples_.num_pcie_lanes_->push_back(bandwidth_info.lanes[bandwidth_info.transfer_rate.current]);
                    } else {
                        // the current index is (somehow) wrong
                        memory_samples_.pcie_transfer_rate_->push_back(0);
                        memory_samples_.num_pcie_lanes_->push_back(0);
                    }
                }
            }

            // retrieve temperature related samples
            {
                if (temperature_samples_.fan_speed_.has_value()) {
                    decltype(temperature_samples_.fan_speed_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_fan_speed_get(device_id_, std::uint32_t{ 0 }, &value));
                    temperature_samples_.fan_speed_->push_back(value);
                }

                if (temperature_samples_.temperature_edge_.has_value()) {
                    decltype(temperature_samples_.temperature_edge_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_edge_->push_back(value);
                }

                if (temperature_samples_.temperature_hotspot_.has_value()) {
                    decltype(temperature_samples_.temperature_hotspot_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_hotspot_->push_back(value);
                }

                if (temperature_samples_.temperature_memory_.has_value()) {
                    decltype(temperature_samples_.temperature_memory_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_memory_->push_back(value);
                }

                if (temperature_samples_.temperature_hbm_0_.has_value()) {
                    decltype(temperature_samples_.temperature_hbm_0_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_0, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_hbm_0_->push_back(value);
                }

                if (temperature_samples_.temperature_hbm_1_.has_value()) {
                    decltype(temperature_samples_.temperature_hbm_1_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_1, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_hbm_1_->push_back(value);
                }

                if (temperature_samples_.temperature_hbm_2_.has_value()) {
                    decltype(temperature_samples_.temperature_hbm_2_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_2, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_hbm_2_->push_back(value);
                }

                if (temperature_samples_.temperature_hbm_3_.has_value()) {
                    decltype(temperature_samples_.temperature_hbm_3_)::value_type::value_type value{};
                    PLSSVM_ROCM_SMI_ERROR_CHECK(rsmi_dev_temp_metric_get(device_id_, RSMI_TEMP_TYPE_HBM_3, RSMI_TEMP_CURRENT, &value));
                    temperature_samples_.temperature_hbm_3_->push_back(value);
                }
            }
        }

        // wait for the sampling interval to pass to retrieve the next sample
        std::this_thread::sleep_for(this->sampling_interval());
    }
}

std::ostream &operator<<(std::ostream &out, const gpu_amd_hardware_sampler &sampler) {
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

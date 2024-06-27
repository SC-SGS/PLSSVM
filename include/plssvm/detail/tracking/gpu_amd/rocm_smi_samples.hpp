/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the samples used with ROCm SMI.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_AMD_ROCM_SMI_SAMPLES_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_AMD_ROCM_SMI_SAMPLES_HPP_
#pragma once

#include "plssvm/detail/tracking/utility.hpp"  // PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER, PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstdint>   // std::uint64_t, std::int64_t, std::uint32_t
#include <iosfwd>    // std::ostream forward declaration
#include <optional>  // std::optional
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all general ROCm SMI hardware samples.
 */
class rocm_smi_general_samples {
    // befriend hardware sampler class
    friend class gpu_amd_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available general hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)  // the name of the device

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, performance_level)          // the performance level: one of rsmi_dev_perf_level_t
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, utilization_gpu)  // the GPU compute utilization in percent
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, utilization_mem)  // the GPU memory utilization in percent
};

/**
 * @brief Output the general @p samples to the given output-stream @p out.
 * @details In contrast to `rocm_smi_general_samples::generate_yaml_string()`, outputs **all** general hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the general hardware samples to
 * @param[in] samples the ROCm SMI general samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const rocm_smi_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all clock related ROCm SMI hardware samples.
 */
class rocm_smi_clock_samples {
    // befriend hardware sampler class
    friend class gpu_amd_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available clock related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_system_min)  // the minimum possible system clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_system_max)  // the maximum possible system clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_socket_min)  // the minimum possible socket clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_socket_max)  // the maximum possible socket clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_memory_min)  // the minimum possible memory clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_memory_max)  // the maximum possible memory clock frequency in Hz

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, clock_system)            // the current system clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, clock_socket)            // the current socket clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, clock_memory)            // the current memory clock frequency in Hz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, overdrive_level)         // the GPU overdrive percentage
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, memory_overdrive_level)  // the GPU's memory overdrive percentage
};

/**
 * @brief Output the clock related @p samples to the given output-stream @p out.
 * @details In contrast to `rocm_smi_clock_samples::generate_yaml_string()`, outputs **all** clock related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the clock related hardware samples to
 * @param[in] samples the ROCm SMI clock related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const rocm_smi_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all power related ROCm SMI hardware samples.
 */
class rocm_smi_power_samples {
    // befriend hardware sampler class
    friend class gpu_amd_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available power related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, power_default_cap)                    // the default power cap, may be different from power cap
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, power_cap)                            // if the GPU draws more power (μW) than the power cap, the GPU may throttle
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, power_type)                             // the type of the power management: either current power draw or average power draw
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::vector<std::string>, available_power_profiles)  // a list of the available power profiles

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, power_usage)                     // the current GPU socket power draw in μW
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, power_total_energy_consumption)  // the total power consumption since the last driver reload in μJ
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::string, power_profile)                     // the current active power profile; one of 'available_power_profiles'
};

/**
 * @brief Output the power related @p samples to the given output-stream @p out.
 * @details In contrast to `rocm_smi_power_samples::generate_yaml_string()`, outputs **all** power related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the power related hardware samples to
 * @param[in] samples the ROCm SMI power related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const rocm_smi_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all memory related ROCm SMI hardware samples.
 */
class rocm_smi_memory_samples {
    // befriend hardware sampler class
    friend class gpu_amd_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available memory related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, memory_total)          // the total available memory in Byte
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, visible_memory_total)  // the total visible available memory in Byte, may be smaller than the total memory
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, min_num_pcie_lanes)    // the minimum number of used PCIe lanes
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, max_num_pcie_lanes)    // the maximum number of used PCIe lanes

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, memory_used)         // the currently used memory in Byte
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, pcie_transfer_rate)  // the current PCIe transfer rate in T/s
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, num_pcie_lanes)      // the number of currently used PCIe lanes
};

/**
 * @brief Output the memory related @p samples to the given output-stream @p out.
 * @details In contrast to `rocm_smi_memory_samples::generate_yaml_string()`, outputs **all** memory related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the memory related hardware samples to
 * @param[in] samples the ROCm SMI memory related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const rocm_smi_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all temperature related ROCm SMI hardware samples.
 */
class rocm_smi_temperature_samples {
    // befriend hardware sampler class
    friend class gpu_amd_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available temperature related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, num_fans)                // the number of fans (if any)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, max_fan_speed)           // the maximum fan speed
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_edge_min)     // the minimum temperature on the GPU's edge temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_edge_max)     // the maximum temperature on the GPU's edge temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hotspot_min)  // the minimum temperature on the GPU's hotspot temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hotspot_max)  // the maximum temperature on the GPU's hotspot temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_memory_min)   // the minimum temperature on the GPU's memory temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_memory_max)   // the maximum temperature on the GPU's memory temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_0_min)    // the minimum temperature on the GPU's HBM0 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_0_max)    // the maximum temperature on the GPU's HBM0 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_1_min)    // the minimum temperature on the GPU's HBM1 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_1_max)    // the maximum temperature on the GPU's HBM1 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_2_min)    // the minimum temperature on the GPU's HBM2 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_2_max)    // the maximum temperature on the GPU's HBM2 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_3_min)    // the minimum temperature on the GPU's HBM3 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hbm_3_max)    // the maximum temperature on the GPU's HBM3 temperature sensor in m°C

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, fan_speed)            // the current fan speed in %
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_edge)     // the current temperature on the GPU's edge temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_hotspot)  // the current temperature on the GPU's hotspot temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_memory)   // the current temperature on the GPU's memory temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_hbm_0)    // the current temperature on the GPU's HBM0 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_hbm_1)    // the current temperature on the GPU's HBM1 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_hbm_2)    // the current temperature on the GPU's HBM2 temperature sensor in m°C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_hbm_3)    // the current temperature on the GPU's HBM3 temperature sensor in m°C
};

/**
 * @brief Output the temperature related @p samples to the given output-stream @p out.
 * @details In contrast to `rocm_smi_temperature_samples::generate_yaml_string()`, outputs **all** temperature related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the temperature related hardware samples to
 * @param[in] samples the ROCm SMI temperature related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const rocm_smi_temperature_samples &samples);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_general_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_clock_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_power_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_memory_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_temperature_samples> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_GPU_AMD_ROCM_SMI_SAMPLES_HPP_

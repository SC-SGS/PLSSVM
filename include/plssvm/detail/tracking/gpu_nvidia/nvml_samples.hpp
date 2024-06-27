/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the samples used with NVML.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_SAMPLES_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_SAMPLES_HPP_
#pragma once

#include "plssvm/detail/tracking/utility.hpp"  // PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER, PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>    // std::ostream forward declaration
#include <optional>  // std::optional
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all general NVML hardware samples.
 */
class nvml_general_samples {
    // befriend hardware sampler class
    friend class gpu_nvidia_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available general hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)        // the name of the device
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(bool, persistence_mode)   // the persistence mode: if true, the driver is always loaded reducing the latency for the first API call
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_cores)  // the number of CUDA cores

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, performance_state)         // the performance state: 0 - 15 where 0 is the maximum performance and 15 the minimum performance
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, utilization_gpu)  // the GPU compute utilization in percent
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, utilization_mem)  // the GPU memory utilization in percent
};

/**
 * @brief Output the general @p samples to the given output-stream @p out.
 * @details In contrast to `nvml_general_samples::generate_yaml_string()`, outputs **all** general hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the general hardware samples to
 * @param[in] samples the NVML general samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const nvml_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all clock related NVML hardware samples.
 */
class nvml_clock_samples {
    // befriend hardware sampler class
    friend class gpu_nvidia_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available clock related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, adaptive_clock_status)  // true if clock boosting is currently enabled
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_graph_min)        // the minimum possible graphics clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_graph_max)        // the maximum possible graphics clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_sm_max)           // the maximum possible SM clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_mem_min)          // the minimum possible memory clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_mem_max)          // the maximum possible memory clock frequency in MHz

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, clock_graph)                  // the current graphics clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, clock_sm)                     // the current SM clock frequency in Mhz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, clock_mem)                    // the current memory clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, clock_throttle_reason)  // the reason the GPU clock throttled (bitmask)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(bool, auto_boosted_clocks)                  // true if the clocks are currently auto boosted
};

/**
 * @brief Output the clock related @p samples to the given output-stream @p out.
 * @details In contrast to `nvml_clock_samples::generate_yaml_string()`, outputs **all** clock related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the clock related hardware samples to
 * @param[in] samples the NVML clock related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const nvml_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all power related NVML hardware samples.
 */
class nvml_power_samples {
    // befriend hardware sampler class
    friend class gpu_nvidia_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available power related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(bool, power_management_mode)           // true if power management algorithms are supported and active
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, power_management_limit)  // if the GPU draws more power (mW) than the power management limit, the GPU may throttle
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, power_enforced_limit)    // the actually enforced power limit, may be different from power management limit if external limiters are set

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, power_state)                                    // the current GPU power state: 0 - 15 where 0 is the maximum power and 15 the minimum power
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, power_usage)                           // the current power draw of the GPU and its related circuity (e.g., memory) in mW
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, power_total_energy_consumption)  // the total power consumption since the last driver reload in mJ
};

/**
 * @brief Output the power related @p samples to the given output-stream @p out.
 * @details In contrast to `nvml_power_samples::generate_yaml_string()`, outputs **all** power related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the power related hardware samples to
 * @param[in] samples the NVML power related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const nvml_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all memory related NVML hardware samples.
 */
class nvml_memory_samples {
    // befriend hardware sampler class
    friend class gpu_nvidia_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available memory related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned long, memory_total)             // the total available memory in Byte
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, pcie_link_max_speed)       // the maximum PCIe link speed in MBPS
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, memory_bus_width)          // the memory bus with in Bit
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, max_pcie_link_generation)  // the current PCIe link generation (e.g., PCIe 4.0, PCIe 5.0, etc)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_free)     // the currently free memory in Byte
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_used)     // the currently used memory in Byte
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, pcie_link_speed)       // the current PCIe link speed in MBPS
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, pcie_link_width)       // the current PCIe link width (e.g., x16, x8, x4, etc)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, pcie_link_generation)  // the current PCIe link generation (may change during runtime to save energy)
};

/**
 * @brief Output the memory related @p samples to the given output-stream @p out.
 * @details In contrast to `nvml_memory_samples::generate_yaml_string()`, outputs **all** memory related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the memory related hardware samples to
 * @param[in] samples the NVML memory related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const nvml_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all temperature related NVML hardware samples.
 */
class nvml_temperature_samples {
    // befriend hardware sampler class
    friend class gpu_nvidia_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available temperature related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_fans)                       // the number of fans (if any)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, min_fan_speed)                  // the minimum fan speed the user can set in %
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, max_fan_speed)                  // the maximum fan speed the user can set in %
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, temperature_threshold_gpu_max)  // the maximum graphics temperature threshold in °C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, temperature_threshold_mem_max)  // the maximum memory temperature threshold in °C

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, fan_speed)        // the current intended fan speed in %
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, temperature_gpu)  // the current GPU temperature in °C
};

/**
 * @brief Output the temperature related @p samples to the given output-stream @p out.
 * @details In contrast to `nvml_temperature_samples::generate_yaml_string()`, outputs **all** temperature related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the temperature related hardware samples to
 * @param[in] samples the NVML temperature related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const nvml_temperature_samples &samples);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_general_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_clock_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_power_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_memory_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_temperature_samples> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_NVML_SAMPLES_HPP_

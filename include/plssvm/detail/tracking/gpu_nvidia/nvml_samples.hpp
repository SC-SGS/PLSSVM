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

class nvml_general_samples {
    friend class gpu_nvidia_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(bool, persistence_mode)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_cores)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, performance_state)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, utilization_gpu)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, utilization_mem)
};

std::ostream &operator<<(std::ostream &out, const nvml_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

class nvml_clock_samples {
    friend class gpu_nvidia_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, adaptive_clock_status)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_graph_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_graph_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_sm_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_mem_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, clock_mem_max)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, clock_graph)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, clock_sm)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, clock_mem)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, clock_throttle_reason)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(bool, auto_boosted_clocks)
};

std::ostream &operator<<(std::ostream &out, const nvml_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

class nvml_power_samples {
    friend class gpu_nvidia_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, power_management_limit)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, power_enforced_limit)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, power_state)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, power_usage)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, power_total_energy_consumption)
};

std::ostream &operator<<(std::ostream &out, const nvml_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

class nvml_memory_samples {
    friend class gpu_nvidia_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned long, memory_total)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, pcie_link_max_speed)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, memory_bus_width)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, max_pcie_link_generation)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_free)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_used)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, pcie_link_speed)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, pcie_link_width)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, pcie_link_generation)
};

std::ostream &operator<<(std::ostream &out, const nvml_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

class nvml_temperature_samples {
    friend class gpu_nvidia_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_fans)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, min_fan_speed)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, max_fan_speed)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, temperature_threshold_gpu_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, temperature_threshold_mem_max)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, fan_speed)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, temperature_gpu)
};

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

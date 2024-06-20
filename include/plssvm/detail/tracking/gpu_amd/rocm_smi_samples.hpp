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

class rocm_smi_general_samples {
    friend class gpu_amd_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, performance_level)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, utilization_gpu)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, utilization_mem)
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

class rocm_smi_clock_samples {
    friend class gpu_amd_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_system_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_system_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_socket_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_socket_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_memory_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, clock_memory_max)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, clock_system)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, clock_socket)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, clock_memory)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, clock_throttle_status)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, overdrive_level)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, memory_overdrive_level)
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

class rocm_smi_power_samples {
    friend class gpu_amd_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, power_default_cap)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, power_cap)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, power_type)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::vector<std::string>, available_power_profiles)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, power_usage)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, power_total_energy_consumption)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::string, power_profile)
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

class rocm_smi_memory_samples {
    friend class gpu_amd_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, memory_total)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, visible_memory_total)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, min_num_pcie_lanes)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, max_num_pcie_lanes)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, memory_used)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, pcie_transfer_rate)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint32_t, num_pcie_lanes)
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

class rocm_smi_temperature_samples {
    friend class gpu_amd_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, num_fans)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint64_t, max_fan_speed)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_edge_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_edge_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hotspot_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_hotspot_max)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_memory_min)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, temperature_memory_max)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, fan_speed)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_edge)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_hotspot)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, temperature_memory)
};

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

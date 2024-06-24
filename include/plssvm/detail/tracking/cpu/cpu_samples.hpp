/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the samples used with lscpu and/or turbostat.
 */

#ifndef PLSSVM_DETAIL_TRACKING_CPU_CPU_SAMPLES_HPP_
#define PLSSVM_DETAIL_TRACKING_CPU_CPU_SAMPLES_HPP_
#pragma once

#include "plssvm/detail/tracking/utility.hpp"  // PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER, PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>         // std::ostream forward declaration
#include <optional>       // std::optional
#include <string>         // std::string
#include <unordered_map>  // std::unordered_map
#include <vector>         // std::vector

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

class cpu_general_samples {
    friend class cpu_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, architecture)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, byte_order)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_threads)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, threads_per_core)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, cores_per_socket)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_sockets)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, numa_nodes)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, vendor_id)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::vector<std::string>, flags)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, busy_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, ipc)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, irq)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, smi)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, poll)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, poll_percent)
};

std::ostream &operator<<(std::ostream &out, const cpu_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

class cpu_clock_samples {
    friend class cpu_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(bool, frequency_boost)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, min_frequency)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, max_frequency)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, average_frequency)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, average_non_idle_frequency)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, time_stamp_counter)
};

std::ostream &operator<<(std::ostream &out, const cpu_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

class cpu_power_samples {
    friend class cpu_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_watt)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, core_watt)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, ram_watt)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_rapl_throttle_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, dram_rapl_throttle_percent)
};

std::ostream &operator<<(std::ostream &out, const cpu_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

class cpu_memory_samples {
    friend class cpu_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l1d_cache)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l1i_cache)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l2_cache)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l3_cache)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned long long, memory_total)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned long long, swap_memory_total)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_used)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_free)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, swap_memory_used)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, swap_memory_free)
};

std::ostream &operator<<(std::ostream &out, const cpu_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

class cpu_temperature_samples {
    friend class cpu_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, core_temperature)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, core_throttle_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_temperature)
};

std::ostream &operator<<(std::ostream &out, const cpu_temperature_samples &samples);

//*************************************************************************************************************************************//
//                                                          gfx (iGPU) samples                                                         //
//*************************************************************************************************************************************//

class cpu_gfx_samples {
    friend class cpu_hardware_sampler;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, gfx_render_state_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, gfx_frequency)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, average_gfx_frequency)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, gfx_state_c0_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, cpu_works_for_gpu_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, gfx_watt)
};

std::ostream &operator<<(std::ostream &out, const cpu_gfx_samples &samples);

//*************************************************************************************************************************************//
//                                                          idle state samples                                                         //
//*************************************************************************************************************************************//

class cpu_idle_states_samples {
    friend class cpu_hardware_sampler;
    using map_type = std::unordered_map<std::string, std::vector<double>>;

  public:
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type, idle_states)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, all_cpus_state_c0_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, any_cpu_state_c0_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, low_power_idle_state_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, system_low_power_idle_state_percent)
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_low_power_idle_state_percent)
};

std::ostream &operator<<(std::ostream &out, const cpu_idle_states_samples &samples);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_general_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_clock_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_power_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_memory_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_temperature_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_gfx_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_idle_states_samples> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_CPU_CPU_SAMPLES_HPP_

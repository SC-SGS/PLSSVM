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

/**
 * @brief Wrapper class for all general CPU hardware samples.
 */
class cpu_general_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available general hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, architecture)        // the CPU architecture (e.g., x86_64)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, byte_order)          // the byte order (e.g., little/big endian)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_threads)        // the number of threads of the CPU(s) including potential hyper-threads
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, threads_per_core)   // the number of hyper-threads per core
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, cores_per_socket)   // the number of physical cores per socket
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, num_sockets)        // the number of sockets
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned int, numa_nodes)         // the number of NUMA nodes
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, vendor_id)           // the vendor ID (e.g. GenuineIntel)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)                // the name of the CPU
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::vector<std::string>, flags)  // potential CPU flags (e.g., sse4_1, avx, avx, etc)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, busy_percent)  // the percent the CPU was busy doing work
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, ipc)           // the instructions-per-cycle count
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, irq)     // the number of interrupts
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, smi)     // the number of system management interrupts
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, poll)    // the number of times the CPU was in the polling state
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, poll_percent)  // the percent of the CPU was in the polling state
};

/**
 * @brief Output the general @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_general_samples::generate_yaml_string()`, outputs **all** general hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the general hardware samples to
 * @param[in] samples the CPU general samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all clock related CPU hardware samples.
 */
class cpu_clock_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available clock related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(bool, frequency_boost)  // true if frequency boosting is enabled
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, min_frequency)  // the minimum possible CPU frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, max_frequency)  // the maximum possible CPU frequency in MHz

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, average_frequency)           // the average CPU frequency in MHz including idle cores
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, average_non_idle_frequency)  // the average CPU frequency in MHz excluding idle cores
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, time_stamp_counter)          // the time stamp counter
};

/**
 * @brief Output the clock related @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_clock_samples::generate_yaml_string()`, outputs **all** clock related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the clock related hardware samples to
 * @param[in] samples the CPU clock related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all power related CPU hardware samples.
 */
class cpu_power_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available power related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_watt)                   // the currently consumed power of the package of the CPU in W
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, core_watt)                      // the currently consumed power of the core part of the CPU in W
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, ram_watt)                       // the currently consumed power of the RAM part of the CPU in W
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_rapl_throttle_percent)  // the percent of time the package throttled due to RAPL limiters
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, dram_rapl_throttle_percent)     // the percent of time the DRAM throttled due to RAPL limiters
};

/**
 * @brief Output the power related @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_power_samples::generate_yaml_string()`, outputs **all** power related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the power related hardware samples to
 * @param[in] samples the CPU power related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all memory related CPU hardware samples.
 */
class cpu_memory_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available memory related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l1d_cache)                 // the size of the L1 data cache
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l1i_cache)                 // the size of the L1 instruction cache
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l2_cache)                  // the size of the L2 cache
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, l3_cache)                  // the size of the L2 cache
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned long long, memory_total)       // the total available memory in Byte
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(unsigned long long, swap_memory_total)  // the total available swap memory in Byte

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_used)       // the currently used memory in Byte
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, memory_free)       // the currently free memory in Byte
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, swap_memory_used)  // the currently used swap memory in Byte
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned long long, swap_memory_free)  // the currently free swap memory in Byte
};

/**
 * @brief Output the memory related @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_memory_samples::generate_yaml_string()`, outputs **all** memory related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the memory related hardware samples to
 * @param[in] samples the CPU memory related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all temperature related CPU hardware samples.
 */
class cpu_temperature_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available temperature related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, core_temperature)  // the current temperature of the core part of the CPU in °C
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, core_throttle_percent)   // the percent of time the CPU has throttled
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_temperature)     // the current temperature of the whole package in °C
};

/**
 * @brief Output the temperature related @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_temperature_samples::generate_yaml_string()`, outputs **all** temperature related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the temperature related hardware samples to
 * @param[in] samples the CPU temperature related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_temperature_samples &samples);

//*************************************************************************************************************************************//
//                                                          gfx (iGPU) samples                                                         //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all gfx (iGPU) related CPU hardware samples.
 */
class cpu_gfx_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available gfx (iGPU) related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, gfx_render_state_percent)  // the percent of time the iGPU was in the render state
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(unsigned int, gfx_frequency)             // the current iGPU power consumption in W
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, average_gfx_frequency)           // the average iGPU frequency in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, gfx_state_c0_percent)            // the percent of the time the iGPU was in the c0 state
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, cpu_works_for_gpu_percent)       // the percent of time the CPU was doing work for the iGPU
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, gfx_watt)                        // the currently consumed power of the iGPU of the CPU in W
};

/**
 * @brief Output the gfx (iGPU) related @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_gfx_samples::generate_yaml_string()`, outputs **all** gfx (iGPU) related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the gfx (iGPU) related hardware samples to
 * @param[in] samples the CPU gfx (iGPU) related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_gfx_samples &samples);

//*************************************************************************************************************************************//
//                                                          idle state samples                                                         //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all idle state related CPU hardware samples.
 */
class cpu_idle_states_samples {
    // befriend hardware sampler class
    friend class cpu_hardware_sampler;
    /// The map type used for the idle state samples that are categorized using the regular expressions.
    using map_type = std::unordered_map<std::string, std::vector<double>>;

  public:
    /**
     * @brief Assemble the YAML string containing all available idle state related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type, idle_states)                            // the map of additional CPU idle states
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, all_cpus_state_c0_percent)             // the percent of time all CPUs were in idle state c0
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, any_cpu_state_c0_percent)              // the percent of time any CPU was in the idle state c0
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, low_power_idle_state_percent)          // the percent of time the CPUs was in the low power idle state
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, system_low_power_idle_state_percent)   // the percent of time the CPU was in the system low power idle state
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, package_low_power_idle_state_percent)  // the percent of time the CPU was in the package low power idle state
};

/**
 * @brief Output the idle state related @p samples to the given output-stream @p out.
 * @details In contrast to `cpu_idle_states_samples::generate_yaml_string()`, outputs **all** idle state related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the idle state related hardware samples to
 * @param[in] samples the CPU idle state related samples
 * @return the output-stream
 */
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

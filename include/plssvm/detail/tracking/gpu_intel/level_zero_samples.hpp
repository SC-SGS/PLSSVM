/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the samples used with Level Zero.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_SAMPLES_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_SAMPLES_HPP_
#pragma once

#include "plssvm/detail/tracking/utility.hpp"  // PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER, PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstdint>        // std::uint64_t, std::int32_t
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
 * @brief Wrapper class for all general Level Zero hardware samples.
 */
class level_zero_general_samples {
    // befriend hardware sampler class
    friend class gpu_intel_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available general hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, name)                  // the model name of the device
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::string, standby_mode)          // the enabled standby mode (power saving or never)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, num_threads_per_eu)  // the number of threads per EU unit
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::uint32_t, eu_simd_width)       // the physical EU unit SIMD width
};

/**
 * @brief Output the general @p samples to the given output-stream @p out.
 * @details In contrast to `level_zero_general_samples::generate_yaml_string()`, outputs **all** general hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the general hardware samples to
 * @param[in] samples the Level Zero general samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const level_zero_general_samples &samples);

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all clock related Level Zero hardware samples.
 */
class level_zero_clock_samples {
    // befriend hardware sampler class
    friend class gpu_intel_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available clock related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, clock_gpu_min)                      // the minimum possible GPU clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, clock_gpu_max)                      // the maximum possible GPU clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::vector<double>, available_clocks_gpu)  // the available GPU clock frequencies in MHz (slowest to fastest)
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, clock_mem_min)                      // the minimum possible memory clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, clock_mem_max)                      // the maximum possible memory clock frequency in MHz
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::vector<double>, available_clocks_mem)  // the available memory clock frequencies in MHz (slowest to fastest)

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, tdp_frequency_limit_gpu)  // the current maximum allowed GPU frequency based on the TDP limit in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, clock_gpu)                // the current GPU frequency in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, throttle_reason_gpu)         // the current GPU frequency throttle reason
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, tdp_frequency_limit_mem)  // the current maximum allowed memory frequency based on the TDP limit in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(double, clock_mem)                // the current memory frequency in MHz
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(int, throttle_reason_mem)         // the current memory frequency throttle reason
};

/**
 * @brief Output the clock related @p samples to the given output-stream @p out.
 * @details In contrast to `level_zero_clock_samples::generate_yaml_string()`, outputs **all** clock related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the clock related hardware samples to
 * @param[in] samples the Level Zero clock related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const level_zero_clock_samples &samples);

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all power related Level Zero hardware samples.
 */
class level_zero_power_samples {
    // befriend hardware sampler class
    friend class gpu_intel_hardware_sampler;

  public:
    /**
     * @brief Assemble the YAML string containing all available power related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(bool, energy_threshold_enabled)  // true if the energy threshold is enabled
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(double, energy_threshold)        // the energy threshold in J

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::uint64_t, power_total_energy_consumption)  // the total power consumption since the last driver reload in mJ
};

/**
 * @brief Output the power related @p samples to the given output-stream @p out.
 * @details In contrast to `level_zero_power_samples::generate_yaml_string()`, outputs **all** power related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the power related hardware samples to
 * @param[in] samples the Level Zero power related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const level_zero_power_samples &samples);

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all memory related Level Zero hardware samples.
 */
class level_zero_memory_samples {
    // befriend hardware sampler class
    friend class gpu_intel_hardware_sampler;

    /**
     * @brief The map type used if the number of potential Level Zero domains is unknown at compile time.
     * @tparam T the mapped type
     */
    template <typename T>
    using map_type = std::unordered_map<std::string, T>;

  public:
    /**
     * @brief Assemble the YAML string containing all available memory related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::uint64_t>, memory_total)              // the total memory size of the different memory modules in Bytes
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::uint64_t>, allocatable_memory_total)  // the total allocatable memory size of the different memory modules in Bytes
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int64_t, pcie_link_max_speed)                  // the maximum PCIe bandwidth in bytes/sec
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int32_t, pcie_max_width)                       // the PCIe lane width
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(std::int32_t, max_pcie_link_generation)             // the PCIe generation
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::int32_t>, bus_width)                  // the bus width of the different memory modules
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::int32_t>, num_channels)               // the number of memory channels of the different memory modules
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::string>, location)                    // the location of the different memory modules (system or device)

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::vector<std::uint64_t>>, memory_free)  // the currently free memory of the different memory modules in Bytes
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int64_t, pcie_link_speed)                   // the current PCIe bandwidth in bytes/sec
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int32_t, pcie_link_width)                   // the current PCIe lane width
    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int32_t, pcie_link_generation)              // the current PCIe generation
};

/**
 * @brief Output the memory related @p samples to the given output-stream @p out.
 * @details In contrast to `level_zero_memory_samples::generate_yaml_string()`, outputs **all** memory related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the memory related hardware samples to
 * @param[in] samples the Level Zero memory related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const level_zero_memory_samples &samples);

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

/**
 * @brief Wrapper class for all temperature related Level Zero hardware samples.
 */
class level_zero_temperature_samples {
    // befriend hardware sampler class
    friend class gpu_intel_hardware_sampler;

    /**
     * @brief The map type used if the number of potential Level Zero domains is unknown at compile time.
     * @tparam T the mapped type
     */
    template <typename T>
    using map_type = std::unordered_map<std::string, T>;

  public:
    /**
     * @brief Assemble the YAML string containing all available temperature related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;

    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<double>, temperature_max)  // the maximum temperature for the sensor in °C

    PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(std::int32_t, temperature_psu)            // the temperature of the PSU in °C
    PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(map_type<std::vector<double>>, temperature)  // the current temperature for the sensor in °C
};

/**
 * @brief Output the temperature related @p samples to the given output-stream @p out.
 * @details In contrast to `level_zero_temperature_samples::generate_yaml_string()`, outputs **all** temperature related hardware samples, even if not supported by the current device (default initialized value).
 * @param[in,out] out the output-stream to write the temperature related hardware samples to
 * @param[in] samples the Level Zero temperature related samples
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const level_zero_temperature_samples &samples);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::level_zero_general_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::level_zero_clock_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::level_zero_power_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::level_zero_memory_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::level_zero_temperature_samples> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_GPU_INTEL_LEVEL_ZERO_SAMPLES_HPP_

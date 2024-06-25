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

#include <iosfwd>    // std::ostream forward declaration
#include <optional>  // std::optional
#include <string>    // std::string
#include <vector>    // std::vector

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

  public:
    /**
     * @brief Assemble the YAML string containing all available memory related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;
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

  public:
    /**
     * @brief Assemble the YAML string containing all available temperature related hardware samples.
     * @details Hardware samples that are not supported by the current device are omitted in the YAML output.
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string generate_yaml_string() const;
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

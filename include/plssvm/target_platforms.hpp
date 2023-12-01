/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible target platforms.
 *        Can also include targets not available on the current target platform.
 */

#ifndef PLSSVM_TARGET_PLATFORMS_HPP_
#define PLSSVM_TARGET_PLATFORMS_HPP_
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm {

/**
 * @brief Enum class for all possible targets.
 */
enum class target_platform {
    /** The default target with respect to the used backend type. Checks for available devices in the following order: NVIDIA GPUs -> AMD GPUs -> Intel GPUs -> CPUs. */
    automatic,
    /** Target CPUs only (Intel, AMD, IBM, ...). */
    cpu,
    /** Target GPUs from NVIDIA. */
    gpu_nvidia,
    /** Target GPUs from AMD. */
    gpu_amd,
    /** Target GPUs from Intel. */
    gpu_intel
};

/**
 * @brief Return a list of all currently available target platforms.
 * @details Only target platforms that where requested during the CMake configuration are available.
 * @return the available target platforms (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<target_platform> list_available_target_platforms();

/**
 * @brief Return the default target platform given the available devices in @p platform_device_list.
 * @details Does not take the currently available backends into account!
 * @param[in] platform_device_list list of target_platforms found in the current setup; if no target_platforms are provided, queries all available target_platforms
 * @return the default target_platform (`[[nodiscard]]`)
 */
[[nodiscard]] target_platform determine_default_target_platform(const std::vector<target_platform> &platform_device_list = list_available_target_platforms());

/**
 * @brief Output the @p target platform to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the target platform to
 * @param[in] target the target platform
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, target_platform target);

/**
 * @brief Use the input-stream @p in to initialize the @p target platform.
 * @param[in,out] in input-stream to extract the target platform from
 * @param[in] target the target platform
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, target_platform &target);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::target_platform> : fmt::ostream_formatter {};

#endif  // PLSSVM_TARGET_PLATFORMS_HPP_
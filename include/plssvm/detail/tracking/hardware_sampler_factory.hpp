/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a factory function for the hardware samplers.
 */

#ifndef PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_
#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_
#pragma once

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform

#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

/**
 * @brief Create a hardware sampler for device @p device_id on the @p target platform.
 * @param[in] target the target platform to create a hardware sampler for
 * @param[in] device_id the device for which to create a hardware sampler for
 * @param[in] sampling_interval the sampling interval to use; default value determined during CMake configuration
 * @return the hardware sampler (`[[nodiscard]]`)
 */
[[nodiscard]] std::unique_ptr<hardware_sampler> make_hardware_sampler(target_platform target, std::size_t device_id, std::chrono::milliseconds sampling_interval);

/**
 * @brief Create @p num_devices many hardware samplers for the @p target platform.
 * @details If available, the CPU is **always** sampled regardless the @p target platform. If the @p target is `plssvm::target_platform::cpu`, only one hardware sampler is created.
 * @param[in] target the target platform that should be sampled
 * @param[in] num_devices the number of devices to sample
 * @param[in] sampling_interval the sampling interval to use; default value determined during CMake configuration
 * @throws plssvm::hardware_sampling_exception all exceptions thrown in the `plssvm::detail::tracking::make_hardware_sampler()` function
 * @throws plssvm::hardware_sampling_exception if the number of devices is zero
 * @return all created hardware samplers (possible sampling different targets) (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<std::unique_ptr<hardware_sampler>> create_hardware_sampler(target_platform target, std::size_t num_devices, std::chrono::milliseconds sampling_interval);

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_

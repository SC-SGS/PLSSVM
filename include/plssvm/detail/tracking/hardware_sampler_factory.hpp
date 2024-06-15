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

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform

#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

[[nodiscard]] std::unique_ptr<hardware_sampler> make_hardware_sampler(target_platform target, std::size_t device_id, std::chrono::milliseconds sampling_interval);

[[nodiscard]] std::vector<std::unique_ptr<hardware_sampler>> create_hardware_sampler(target_platform target, std::size_t num_devices, std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_FACTORY_HPP_

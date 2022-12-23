/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the SYCL backend using DPC++ as SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DPCPP_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_SYCL_DPCPP_DETAIL_UTILITY_HPP_
#pragma once

#include "plssvm/backends/SYCL/DPCPP/detail/queue.hpp"  // plssvm::dpcpp::detail::queue (PImpl)
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform

#include <utility>  // std::pair
#include <vector>   // std::vector

namespace plssvm::dpcpp::detail {

/**
 * @brief Returns the list devices matching the target platform @p target and the actually used target platform
 *        (only interesting if the provided @p target was automatic).
 * @details If the selected target platform is `plssvm::target_platform::automatic` the selector tries to find devices in the following order:
 *          1. NVIDIA GPUs
 *          2. AMD GPUs
 *          3. Intel GPUs
 *          4. CPUs
 *
 * @param[in] target the target platform for which the devices must match
 * @return the devices and used target platform (`[[nodiscard]]`)
 */
[[nodiscard]] std::pair<std::vector<queue>, target_platform> get_device_list(target_platform target);

/**
 * @brief Wait for the compute device associated with @p q to finish.
 * @param[in] q the SYCL queue to synchronize
 */
void device_synchronize(const queue &q);

/**
 * @brief Get the default SYCL queue.
 * @details Only used in the tests, but **must** be defined and implemented here!
 * @return the default queue (`[[nodiscard]]`)
 */
[[nodiscard]] queue get_default_queue();

}  // namespace plssvm::dpcpp::detail

#endif  // PLSSVM_BACKENDS_SYCL_DPCPP_DETAIL_UTILITY_HPP_
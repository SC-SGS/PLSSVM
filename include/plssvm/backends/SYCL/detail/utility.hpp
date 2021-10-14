/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the SYCL backend.
 */

#pragma once

#include "plssvm/target_platform.hpp"  // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::queue

#include <vector>  // std::vector

namespace plssvm::sycl::detail {

/**
 * @brief Returns the list devices matching the target platform @p target.
 * @details If the selected target platform is `plssvm::target_platform::automatic` the selector tries to find devices in the following order:
 *          1. NVIDIA GPUs
 *          2. AMD GPUs
 *          3. Intel GPUs
 *          4. CPUs
 * @param[in] target the target platform for which the devices must match
 * @return the devices (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<::sycl::queue> get_device_list(target_platform target);
/**
 * @brief Wait for the compute device associated with @p queue to finish.
 * @param[in] queue the SYCL queue to synchronize
 */
void device_synchronize(::sycl::queue &queue);

}  // namespace plssvm::sycl::detail
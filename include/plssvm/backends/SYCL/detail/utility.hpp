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

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/constants.hpp" // forward declaration and namespace alias
#include "plssvm/target_platforms.hpp"                                             // plssvm::target_platform

#include <memory>   // std::unique_ptr
#include <utility>  // std::pair
#include <vector>   // std::vector

namespace plssvm {

namespace @PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@ {

namespace detail {

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
[[nodiscard]] std::pair<std::vector<std::unique_ptr<detail::sycl::queue>>, target_platform> get_device_list(target_platform target);
/**
 * @brief Wait for the compute device associated with @p queue to finish.
 * @param[in] queue the SYCL queue to synchronize
 */
void device_synchronize(detail::sycl::queue &queue);

}  // namespace detail
}  // namespace @PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@
}  // namespace plssvm
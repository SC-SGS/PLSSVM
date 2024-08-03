/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the SYCL backend using AdaptiveCpp as SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_ADAPTIVECPP_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_SYCL_ADAPTIVECPP_DETAIL_UTILITY_HPP_
#pragma once

#include "plssvm/backends/execution_range.hpp"                // plssvm::detail::dim_type
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/queue.hpp"  // plssvm::adaptivecpp::detail::queue (PImpl)
#include "plssvm/target_platforms.hpp"                        // plssvm::target_platform

#include "sycl/sycl.hpp"  // sycl::range

#include <string>   // std::string
#include <utility>  // std::pair
#include <vector>   // std::vector

namespace plssvm::adaptivecpp::detail {

/**
 * @brief Convert a `plssvm::detail::dim_type` to a SYCL native range.
 * @tparam I the number of dimensions in the SYCL range
 * @param[in] dims the dimensional value to convert
 * @note Inverts the dimensions to account for SYCL's different iteration range!
 * @return the native SYCL range type (`[[nodiscard]]`)
 */
template <std::size_t I>
[[nodiscard]] ::sycl::range<I> dim_type_to_native(const ::plssvm::detail::dim_type &dims) {
    // note: invert order to account for SYCL's different iteration range
    if constexpr (I == 1) {
        return ::sycl::range<I>{ static_cast<std::size_t>(dims.x) };
    } else if constexpr (I == 2) {
        return ::sycl::range<I>{ static_cast<std::size_t>(dims.y), static_cast<std::size_t>(dims.x) };
    } else if constexpr (I == 3) {
        return ::sycl::range<I>{ static_cast<std::size_t>(dims.z), static_cast<std::size_t>(dims.y), static_cast<std::size_t>(dims.x) };
    } else {
        static_assert(I != I, "Invalid number of native sycl::range dimension!");
    }
}

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

/**
 * @brief Return the short AdaptiveCpp version, i.e., major.minor.patch.
 * @return the short AdaptiveCpp version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_adaptivecpp_version_short();
/**
 * @brief Return the full AdaptiveCpp version including git information.
 * @return the full AdaptiveCpp version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_adaptivecpp_version();

}  // namespace plssvm::adaptivecpp::detail

#endif  // PLSSVM_BACKENDS_SYCL_ADAPTIVECPP_DETAIL_UTILITY_HPP_

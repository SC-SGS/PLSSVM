/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a point in time used in the hardware samplers.
 */

#ifndef PLSSVM_DETAIL_TRACKING_TIME_POINT_HPP_
#define PLSSVM_DETAIL_TRACKING_TIME_POINT_HPP_

#include <chrono>  // std::chrono::{steady_clock, duration_cast, milliseconds}

namespace plssvm::detail::tracking {

using clock_type = std::chrono::steady_clock;
using time_point_type = typename clock_type ::time_point;

inline unsigned long long time_point_to_epoch(const time_point_type time) {
    // convert time_point to epoch in milliseconds
    return std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count();
}

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_TIME_POINT_HPP_

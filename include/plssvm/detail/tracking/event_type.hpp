/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an event type.
 */

#ifndef PLSSVM_DETAIL_TRACKING_EVENT_HPP_
#define PLSSVM_DETAIL_TRACKING_EVENT_HPP_

#include "plssvm/detail/tracking/time_point.hpp"  // plssvm::detail::tracking::time_point_type

#include <string>  // std::string

namespace plssvm::detail::tracking {

struct event_type {
    time_point_type time;
    std::string name;
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_EVENT_HPP_

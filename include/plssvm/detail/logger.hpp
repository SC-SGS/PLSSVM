/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a simple logging function.
 * @details Also used for the plssvm::detail::performance_tracker.
 */

#ifndef PLSSVM_DETAIL_LOGGER_HPP_
#define PLSSVM_DETAIL_LOGGER_HPP_
#pragma once

#include "plssvm/constants.hpp"                   // plssvm::verbose
#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::is_tracking_entry_v, PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // format types with an operator<< overload

#include <iostream>     // std::cout
#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm::detail {

/**
 * @breif Output the message @p msg filling the {fmt} like placeholders with @p args to the standard output stream.
 * @details If a value in @p Args is of type plssvm::detail::tracking_entry and performance tracking is enabled,
 *          this is also added to the plssvm::detail::performance_tracker.
 * @tparam Args the types of the placeholder values
 * @param[in] msg the message to print on the standard output stream if requested (i.e., plssvm::verbose is `true`)
 * @param[in] args the values to fill the {fmt}-like placeholders in @p msg
 */
template <typename... Args>
void log(const std::string_view msg, Args &&...args) {
    // output message only if the verbose flag is set to true
    if (verbose) {
        std::cout << fmt::format(msg, args...);
    }

    // if performance tracking has been enabled, add tracking entries
    ([](auto &&arg) {
        if constexpr (detail::is_tracking_entry_v<decltype(arg)>) {
            PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(std::forward<decltype(arg)>(arg));
        }
    }(std::forward<Args>(args)),
     ...);
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LOGGER_HPP_

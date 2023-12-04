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

#ifndef PLSSVM_DETAIL_LOGGING_HPP_
#define PLSSVM_DETAIL_LOGGING_HPP_
#pragma once

#include "plssvm/verbosity_levels.hpp"  // plssvm::verbosity_level, plssvm::verbosity

#if !defined(PLSSVM_LOG_WITHOUT_PERFORMANCE_TRACKING)
    #include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::is_tracking_entry_v, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#endif

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/color.h"   // fmt::fg, fmt::color
#include "fmt/format.h"  // fmt::format

#include <iostream>     // std::cout, std::clog
#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm::detail {

/**
 * @breif Output the message @p msg filling the {fmt} like placeholders with @p args to the standard output stream.
 * @details If a value in @p Args is of type plssvm::detail::tracking_entry and performance tracking is enabled,
 *          this is also added to the plssvm::detail::performance_tracker.
 *          Only logs the message if the verbosity level matches the `plssvm::verbosity` level.
 * @tparam Args the types of the placeholder values
 * @param[in] verb the verbosity level of the message to log; must match the `plssvm::verbosity` level to log the message
 * @param[in] msg the message to print on the standard output stream if requested (i.e., plssvm::verbose is `true`)
 * @param[in] args the values to fill the {fmt}-like placeholders in @p msg
 */
template <typename... Args>
void log(const verbosity_level verb, const std::string_view msg, Args &&...args) {
    // if the verbosity level is quiet, nothing is logged
    // otherwise verb must contain the bit-flag set by plssvm::verbosity
    if (verbosity != verbosity_level::quiet && (verb & verbosity) != verbosity_level::quiet) {
        if ((verb & verbosity_level::warning) != verbosity_level::quiet) {
            std::clog << fmt::format(fmt::fg(fmt::color::orange), msg, args...);
        } else {
            std::cout << fmt::format(msg, args...);
        }
    }

#if !defined(PLSSVM_LOG_WITHOUT_PERFORMANCE_TRACKING)
    // if performance tracking has been enabled, add tracking entries
    ([](auto &&arg) {
        if constexpr (detail::is_tracking_entry_v<decltype(arg)>) {
            PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(std::forward<decltype(arg)>(arg));
        }
    }(std::forward<Args>(args)),
     ...);
#endif
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LOGGING_HPP_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a simple logging function without MPI support. Wrapper that disables performance tracking due to circular dependencies.
 */

#ifndef PLSSVM_DETAIL_LOGGING_LOG_UNTRACKED_HPP_
#define PLSSVM_DETAIL_LOGGING_LOG_UNTRACKED_HPP_
#pragma once

#include "plssvm/verbosity_levels.hpp"  // plssvm::verbosity_level, plssvm::verbosity, bitwise-operators on plssvm::verbosity_level

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/color.h"   // fmt::fg, fmt::color
#include "fmt/format.h"  // fmt::format, fmt::runtime

#include <iostream>     // std::cout, std::clog, std::flush
#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm::detail {

/**
 * @brief Output the message @p msg filling the {fmt} like placeholders with @p args to the standard output stream.
 * @details Only logs the message if the verbosity level matches the `plssvm::verbosity` level.
 * @tparam Args the types of the placeholder values
 * @param[in] msg_verbosity the verbosity level of the message to log
 * @param[in] msg the message to print on the standard output stream if requested (i.e., `plssvm::verbosity` isn't `plssvm::verbosity_level::quiet`)
 * @param[in] args the values to fill the {fmt}-like placeholders in @p msg
 */
template <typename... Args>
void log_untracked(const verbosity_level msg_verbosity, const std::string_view msg, Args &&...args) {
    // verbosity = the currently active verbosity level
    // msg_verbosity = the verbosity of the current message

    // if the global verbosity or the message verbosity is 'plssvm::verbosity_level::quiet', nothing should be logged
    if (!(verbosity == verbosity_level::quiet || msg_verbosity == verbosity_level::quiet)) {
        // check whether the provided msg_verbosity is contained in the current active verbosity
        if ((verbosity & msg_verbosity) != verbosity_level::quiet || (verbosity == verbosity_level::full && (msg_verbosity & verbosity_level::libsvm) == verbosity_level::quiet)) {
            // check if it is a warning message, if yes, the output will be colored
            if ((msg_verbosity & verbosity_level::warning) != verbosity_level::quiet) {
                std::clog << fmt::format(fmt::fg(fmt::color::orange), fmt::runtime(msg), std::forward<Args>(args)...) << std::flush;
            } else {
                std::cout << fmt::format(fmt::runtime(msg), std::forward<Args>(args)...) << std::flush;
            }
        }
    }
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LOGGING_LOG_UNTRACKED_HPP_

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

#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::is_tracking_entry_v, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/utility.hpp"              // PLSSVM_EXTERN

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>       // std::istream, std::ostream
#include <iostream>     // std::cout
#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm {

/**
 * @brief Enum class for all possible verbosity levels.
 */
enum class verbosity_level {
    /** Nothing is logged to the standard output. */
    quiet = 0b000,
    /** Log the same messages as LIBSVM (used for better LIBSVM conformity). */
    libsvm = 0b001,
    /** Log all messages related to timing information. */
    timing = 0b010,
    /** Log all messages. */
    full = 0b100
};

/// The verbosity level used in the logging function. My be changed by the user.
PLSSVM_EXTERN verbosity_level verbosity;

/**
 * @brief Output the @p verb to the given output-stream @p out.
 * @details If more than one verbosity level is provided, outputs all of them, e.g., "libsvm | timing".
 * @param[in,out] out the output-stream to write the verbosity level to
 * @param[in] verb the verbosity level
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, verbosity_level verb);

/**
 * @brief Use the input-stream @p in to initialize the @p verb level.
 * @details If more than one verbosity level is provided, e.g., "libsvm | timing" returns a bitwise-or of the respective enum values.
 *          If any of the values is "quiet", the result will always be `plssvm::verbosity_level::quiet`.
 * @param[in,out] in input-stream to extract the verbosity level from
 * @param[in] verb the verbosity level
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, verbosity_level &verb);

/**
 * @brief Bitwise-or to set multiple verbosity levels at once for a logging message.
 * @param[in] lhs the first verbosity level
 * @param[in] rhs the second verbosity level
 * @return the logical-or of the two verbosity levels (`[[nodiscard]]`)
 */
[[nodiscard]] verbosity_level operator|(verbosity_level lhs, verbosity_level rhs);
/**
 * @brief Bitwise-or to set multiple verbosity levels at once for a logging message.
 * @param[in] lhs the first verbosity level
 * @param[in] rhs the second verbosity level
 * @return the logical-or of the two verbosity levels
 */
verbosity_level operator|=(verbosity_level &lhs, verbosity_level rhs);

/**
 * @brief Bitwise-and to check verbosity levels for a logging message.
 * @param[in] lhs the first verbosity level
 * @param[in] rhs the second verbosity level
 * @return the logical-and of the two verbosity levels (`[[nodiscard]]`)
 */
[[nodiscard]] verbosity_level operator&(verbosity_level lhs, verbosity_level rhs);
/**
 * @brief Bitwise-and to check verbosity levels for a logging message.
 * @param[in] lhs the first verbosity level
 * @param[in] rhs the second verbosity level
 * @return the logical-and of the two verbosity levels
 */
verbosity_level operator&=(verbosity_level &lhs, verbosity_level rhs);

namespace detail {

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
        std::cout << fmt::format(msg, args...);
    }

    // if performance tracking has been enabled, add tracking entries
    ([](auto &&arg) {
        if constexpr (detail::is_tracking_entry_v<decltype(arg)>) {
            PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(std::forward<decltype(arg)>(arg));
        }
    }(std::forward<Args>(args)),
     ...);
}

}  // namespace detail

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::verbosity_level> : fmt::ostream_formatter {};

#endif  // PLSSVM_DETAIL_LOGGER_HPP_

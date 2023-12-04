/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible verbosity levels.
 */

#ifndef PLSSVM_VERBOSITY_LEVELS_HPP_
#define PLSSVM_VERBOSITY_LEVELS_HPP_

#include "plssvm/detail/utility.hpp"  // PLSSVM_EXTERN

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // std::istream, std::ostream

namespace plssvm {

/**
 * @brief Enum class for all possible verbosity levels.
 */
enum class verbosity_level {
    /** Nothing is logged to the standard output. */
    quiet = 0b0000,
    /** Log the same messages as LIBSVM (used for better LIBSVM conformity). */
    libsvm = 0b0001,
    /** Log all messages related to timing information. */
    timing = 0b0010,
    /** Log all messages related to warnings. */
    warning = 0b0100,
    /** Log all messages (i.e., timing, warning, and additional messages). */
    full = 0b1000
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

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::verbosity_level> : fmt::ostream_formatter {};

#endif  // PLSSVM_VERBOSITY_LEVELS_HPP_

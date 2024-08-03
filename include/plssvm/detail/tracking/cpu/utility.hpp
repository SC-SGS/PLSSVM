/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functionality for the CPU sampler.
 */

#ifndef PLSSVM_DETAIL_TRACKING_CPU_UTILITY_HPP_
#define PLSSVM_DETAIL_TRACKING_CPU_UTILITY_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include "fmt/format.h"  // fmt::format

#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm::detail::tracking {

/**
 * @def PLSSVM_SUBPROCESS_ERROR_CHECK
 * @brief Defines the `PLSSVM_SUBPROCESS_ERROR_CHECK` macro if `PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED` is defined, does nothing otherwise.
 * @details Throws an exception if a subprocess call returns with an error.
 */
#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)
    #define PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_func)                                                                      \
        {                                                                                                                       \
            const int errc = subprocess_func;                                                                                   \
            if (errc != 0) {                                                                                                    \
                throw hardware_sampling_exception{ fmt::format("Error calling subprocess function \"{}\"", #subprocess_func) }; \
            }                                                                                                                   \
        }
#else
    #define PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_func) subprocess_func;
#endif

/**
 * @brief Run a subprocess executing @p cmd_line and returning the stdout and stderr string.
 * @param[in] cmd_line the command line to execute
 * @return the stdout and stderr content encountered during executing @p cmd_line (`[[nodiscard]]`)
 */
[[nodiscard]] std::string run_subprocess(std::string_view cmd_line);

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_CPU_UTILITY_HPP_

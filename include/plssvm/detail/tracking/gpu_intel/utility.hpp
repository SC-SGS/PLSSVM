/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functionality for the Intel GPU sampler.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_INTEL_UTILITY_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_INTEL_UTILITY_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include "fmt/core.h"           // fmt::format
#include "level_zero/ze_api.h"  // Level Zero runtime functions

#include <string_view>  // std::string_view

namespace plssvm::detail::tracking {

/**
 * @brief Given the Level Zero API error @p errc, returns a useful error string.
 * @param[in] errc the Level Zero API error code
 * @return the error string (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view to_result_string(const ze_result_t errc);

/**
 * @def PLSSVM_LEVEL_ZERO_ERROR_CHECK
 * @brief Defines the `PLSSVM_LEVEL_ZERO_ERROR_CHECK` macro if `PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED` is defined, does nothing otherwise.
 * @details Throws an exception if a Level Zero call returns with an error. Additionally outputs a more concrete custom error string.
 */
#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)
    #define PLSSVM_LEVEL_ZERO_ERROR_CHECK(level_zero_func)                                                                                                  \
        {                                                                                                                                                   \
            const ze_result_t errc = level_zero_func;                                                                                                       \
            if (errc != ZE_RESULT_SUCCESS) {                                                                                                                \
                throw hardware_sampling_exception{ fmt::format("Error in Level Zero function call \"{}\": {}", #level_zero_func, to_result_string(errc)) }; \
            }                                                                                                                                               \
        }
#else
    #define PLSSVM_LEVEL_ZERO_ERROR_CHECK(level_zero_func) level_zero_func;
#endif

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_INTEL_UTILITY_HPP_

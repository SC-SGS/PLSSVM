/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functionality for the NVIDIA GPU sampler.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_UTILITY_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_UTILITY_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include "fmt/format.h"  // fmt::format
#include "nvml.h"        // NVML runtime functions

namespace plssvm::detail::tracking {

/**
 * @def PLSSVM_NVML_ERROR_CHECK
 * @brief Defines the `PLSSVM_NVML_ERROR_CHECK` macro if `PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED` is defined, does nothing otherwise.
 * @details Throws an exception if an NVML call returns with an error. Additionally outputs a more concrete error string.
 */
#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)
    #define PLSSVM_NVML_ERROR_CHECK(nvml_func)                                                                                                                              \
        {                                                                                                                                                                   \
            const nvmlReturn_t errc = nvml_func;                                                                                                                            \
            if (errc != NVML_SUCCESS) {                                                                                                                                     \
                throw hardware_sampling_exception{ fmt::format("Error in NVML function call \"{}\": {} ({})", #nvml_func, nvmlErrorString(errc), static_cast<int>(errc)) }; \
            }                                                                                                                                                               \
        }
#else
    #define PLSSVM_NVML_ERROR_CHECK(nvml_func) nvml_func;
#endif

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_NVIDIA_UTILITY_HPP_

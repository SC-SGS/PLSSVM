/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functionality for the AMD GPU sampler.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_AMD_UTILITY_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_AMD_UTILITY_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception

#include "fmt/core.h"           // fmt::format
#include "rocm_smi/rocm_smi.h"  // ROCm SMI runtime functions

namespace plssvm::detail::tracking {

/**
 * @def PLSSVM_ROCM_SMI_ERROR_CHECK
 * @brief Defines the `PLSSVM_ROCM_SMI_ERROR_CHECK` macro if `PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED` is defined, does nothing otherwise.
 * @details Throws an exception if a ROCm SMI call returns with an error. Additionally outputs a more concrete error string if possible.
 */
#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)
    #define PLSSVM_ROCM_SMI_ERROR_CHECK(rocm_smi_func)                                                                                                      \
        {                                                                                                                                                   \
            const rsmi_status_t errc = rocm_smi_func;                                                                                                       \
            if (errc != RSMI_STATUS_SUCCESS) {                                                                                                              \
                const char *error_string;                                                                                                                   \
                const rsmi_status_t ret = rsmi_status_string(errc, &error_string);                                                                          \
                if (ret == RSMI_STATUS_SUCCESS) {                                                                                                           \
                    throw hardware_sampling_exception{ fmt::format("Error in ROCm SMI function call \"{}\": {}", #rocm_smi_func, error_string) };           \
                } else {                                                                                                                                    \
                    throw hardware_sampling_exception{ fmt::format("Error in ROCm SMI function call \"{}\": {}", #rocm_smi_func, static_cast<int>(errc)) }; \
                }                                                                                                                                           \
            }                                                                                                                                               \
        }
#else
    #define PLSSVM_ROCM_SMI_ERROR_CHECK(rocm_smi_func) rocm_smi_func;
#endif

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_GPU_AMD_UTILITY_HPP_

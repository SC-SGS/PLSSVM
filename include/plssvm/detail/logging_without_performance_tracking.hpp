/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a simple logging function. Wrapper that disables performance tracking due to circular dependencies.
 */

#ifndef PLSSVM_DETAIL_LOGGING_WITHOUT_PERFORMANCE_TRACKING_HPP_
#define PLSSVM_DETAIL_LOGGING_WITHOUT_PERFORMANCE_TRACKING_HPP_
#pragma once

#define PLSSVM_LOG_WITHOUT_PERFORMANCE_TRACKING
#include "plssvm/detail/logging.hpp"  // plssvm::detail::log
#undef PLSSVM_LOG_WITHOUT_PERFORMANCE_TRACKING

#endif  // PLSSVM_DETAIL_LOGGING_WITHOUT_PERFORMANCE_TRACKING_HPP_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the stdpar backend.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_
#pragma once

#include <string>  // std::string

namespace plssvm::stdpar::detail {

/**
 * @brief Return the stdpar implementation used.
 * @return the stdpar implementation (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_stdpar_implementation();

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_

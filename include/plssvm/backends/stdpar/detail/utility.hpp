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

// TODO: get the used stdpar implementation?

/**
 * @brief Return the stdpar version used (i.e. the used C++ version).
 * @details Parses the output of `__cplusplus` according to https://en.cppreference.com/w/cpp/preprocessor/replace.
 * @return the C++ standard version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_stdpar_version();

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_

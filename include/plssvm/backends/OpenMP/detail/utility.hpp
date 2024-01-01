/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_OPENMP_DETAIL_UTILITY_HPP_
#pragma once

#include <string>  // std::string

namespace plssvm::openmp::detail {

/**
 * @brief Return the number of used CPU threads in the OpenMP backend.
 * @return the number of used CPU threads (`[[nodiscard]]`)
 */
[[nodiscard]] int get_num_threads();
/**
 * @brief Return the OpenMP version used.
 * @details Parses the output of `_OPENMP` according to https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux.
 * @return the OpenMP version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_openmp_version();

}  // namespace plssvm::openmp::detail

#endif  // PLSSVM_BACKENDS_OPENMP_DETAIL_UTILITY_HPP_

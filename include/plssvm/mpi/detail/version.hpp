/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines some version functions for our optional MPI usage.
 */

#ifndef PLSSVM_MPI_DETAIL_VERSION_HPP_
#define PLSSVM_MPI_DETAIL_VERSION_HPP_

#include <string>  // std::string

namespace plssvm::mpi::detail {

/**
 * @brief Get the used MPI library version.
 * @return the MPI library version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string mpi_library_version();

/**
 * @brief Get the used MPI version.
 * @return the MPI version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string mpi_version();

}  // namespace plssvm::mpi::detail

#endif  // PLSSVM_MPI_DETAIL_VERSION_HPP_

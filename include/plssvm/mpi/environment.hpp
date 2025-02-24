/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines some wrapper functions around MPI specific environment functions.
 * @details These wrapper functions are only conditionally compiled such that MPI is still an **optional** dependency in PLSSVM.
 */

#ifndef PLSSVM_MPI_ENVIRONMENT_HPP_
#define PLSSVM_MPI_ENVIRONMENT_HPP_
#pragma once

namespace plssvm::mpi {

/**
 * @brief Initialize the MPI environment.
 * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does nothing.
 */
void init();
/**
 * @brief Initialize the MPI environment with the provided command line arguments.
 * @param[in,out] argc the number of command line arguments
 * @param[in,out] argv the values of the command line arguments
 */
void init(int &argc, char **argv);

/**
 * @brief Finalize the MPI environment.
 * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does nothing.
 */
void finalize();
/**
 * @brief Abort the MPI environment associated with `MPI_COMM_WORLD`.
 * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does nothing.
 */
void abort_world();

/**
 * @brief Check if the MPI environment has been successfully initialized.
 * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `true`.
 * @return `true` if the environment was successfully initialized, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_initialized();
/**
 * @brief Check if the MPI environment has been successfully finalized.
 * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `true`.
 * @return `true` if the environment was successfully finalized, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_finalized();
/**
 * @brief Check if the MPI environment is currently active, i.e., `init` has already been called, but not `finalize`.
 * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `false`.
 * @return `true` if the environment is currently active, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_active();

/**
 * @brief Returns `true` if the executable was started via `mpirun`.
 * @details Checks for the existence of the environment variables: ``
 * @note Will falsely return `true` if any of the environment variables is explicitly set by the user!
 * @return `true` if the executable was started via `mpirun`, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_executed_via_mpirun();

}  // namespace plssvm::mpi

#endif  // PLSSVM_MPI_ENVIRONMENT_HPP_

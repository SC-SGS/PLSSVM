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

void init();
void init(int &argc, char **argv);
void finalize();

[[nodiscard]] bool is_initialized();
[[nodiscard]] bool is_finalized();

}  // namespace plssvm::mpi

#endif  // PLSSVM_MPI_ENVIRONMENT_HPP_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions to gather MPI rank specific information on the main rank and print the out.
 */

#ifndef PLSSVM_MPI_DETAIL_INFORMATION_HPP_
#define PLSSVM_MPI_DETAIL_INFORMATION_HPP_
#pragma once

#include "plssvm/backend_types.hpp"     // plssvm::backend_type
#include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator
#include "plssvm/solver_types.hpp"      // plssvm::solver_type
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::mpi::detail {

/**
 * @brief Communicate the @p solver from each MPI rank in @p comm to @p comm's main rank and outputs the result to the console.
 * @details Only outputs the content on the main MPI rank!
 * @param[in] comm the communicator to gather the solver information from
 * @param[in] rank_solver the solver type used on the current MPI rank, gathered on the main MPI rank
 */
void gather_and_print_solver_information(const communicator &comm, solver_type rank_solver);

/**
 * @brief Communicate the CSVM information, including the used backend, target platform, and device names, from each MPI rank in @p comm to @p comm's main rank and outputs the result to the console.
 * @param[in] comm the communicator to gather the solver information from
 * @param[in] rank_backend the backend used on the current MPI rank, gathered on the main MPI rank
 * @param[in] rank_target the target platform used on the current MPI rank, gathered on the main MPI rank
 * @param[in] rank_devices the device (names) used on the current MPI rank, gathered on the main MPI rank
 * @param[in] additional_info optional additional information used on the current MPI rank, gathered on the main MPI rank
 */
void gather_and_print_csvm_information(const communicator &comm, backend_type rank_backend, target_platform rank_target, const std::vector<std::string> &rank_devices, const std::optional<std::string> &additional_info = std::nullopt);

}  // namespace plssvm::mpi::detail

#endif  // PLSSVM_MPI_DETAIL_INFORMATION_HPP_

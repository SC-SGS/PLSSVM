/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a simple logging function with MPI support. Wrapper that disables performance tracking due to circular dependencies.
 */

#ifndef PLSSVM_DETAIL_LOGGING_MPI_LOG_UNTRACKED_HPP_
#define PLSSVM_DETAIL_LOGGING_MPI_LOG_UNTRACKED_HPP_
#pragma once

#include "plssvm/detail/logging/log_untracked.hpp"  // plssvm::detail::log_untracked
#include "plssvm/mpi/communicator.hpp"              // plssvm::mpi::communicator
#include "plssvm/verbosity_levels.hpp"              // plssvm::verbosity_level, plssvm::verbosity, bitwise-operators on plssvm::verbosity_level

#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm::detail {

/**
 * @brief Output the message @p msg filling the {fmt} like placeholders with @p args to the standard output stream if @p comm represents the current main MPI rank.
 * @details Only logs the message if the verbosity level matches the `plssvm::verbosity` level.
 * @tparam Args the types of the placeholder values
 * @param[in] verb the verbosity level of the message to log; must match the `plssvm::verbosity` level to log the message
 * @param[in] comm the used MPI communicator
 * @param[in] msg the message to print on the standard output stream if requested (i.e., `plssvm::verbosity` isn't `plssvm::verbosity_level::quiet`)
 * @param[in] args the values to fill the {fmt}-like placeholders in @p msg
 */
template <typename... Args>
void log_untracked(const verbosity_level verb, const mpi::communicator &comm, const std::string_view msg, Args &&...args) {
    if (comm.is_main_rank()) {
        // only print on the main MPI rank
        log_untracked(verb, msg, std::forward<Args>(args)...);
    }
    // nothing to do on other MPI ranks
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LOGGING_LOG_UNTRACKED_HPP_

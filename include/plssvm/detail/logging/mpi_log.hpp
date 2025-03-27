/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a simple logging function with MPI support. Additionally, depends on the `plssvm::detail::performance_tracker`.
 */

#ifndef PLSSVM_DETAIL_LOGGING_MPI_LOG_HPP_
#define PLSSVM_DETAIL_LOGGING_MPI_LOG_HPP_
#pragma once

#include "plssvm/detail/logging/log.hpp"      // plssvm::detail::log
#include "plssvm/mpi/communicator.hpp"        // plssvm::mpi::communicator
#include "plssvm/verbosity_levels.hpp"        // plssvm::verbosity_level, plssvm::verbosity, bitwise-operators on plssvm::verbosity_level

#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm::detail {

/**
 * @brief Output the message @p msg filling the {fmt} like placeholders with @p args to the standard output stream if @p comm represents the current main MPI rank.
 * @details If a value in @p args is of type plssvm::detail::tracking_entry and performance tracking is enabled,
 *          this is also added to the `plssvm::detail::performance_tracker`.
 *          Only logs the message if the verbosity level matches the `plssvm::verbosity` level.
 * @tparam Args the types of the placeholder values
 * @param[in] msg_verbosity the verbosity level of the message to log
 * @param[in] comm the used MPI communicator
 * @param[in] msg the message to print on the standard output stream if requested (i.e., `plssvm::verbosity` isn't `plssvm::verbosity_level::quiet`)
 * @param[in] args the values to fill the {fmt}-like placeholders in @p msg
 */
template <typename... Args>
void log(const verbosity_level msg_verbosity, const mpi::communicator &comm, const std::string_view msg, Args &&...args) {
    if (comm.is_main_rank()) {
        // only print on the main MPI rank
        log(msg_verbosity, msg, std::forward<Args>(args)...);
    } else {
        // set output to quiet otherwise (since all MPI ranks should track their args)
        log(verbosity_level::quiet, msg, std::forward<Args>(args)...);
    }
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LOGGING_MPI_LOG_HPP_

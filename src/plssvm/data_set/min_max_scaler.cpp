/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set/min_max_scaler.hpp"

#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"                // plssvm::detail::io::file_reader
#include "plssvm/detail/io/scaling_factors_parsing.hpp"    // plssvm::detail::io::parse_scaling_factors
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::min_max_scaler_exception
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format

#include <chrono>   // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <string>   // std::string
#include <tuple>    // std::tie
#include <utility>  // std::move, std::make_pair

namespace plssvm {

min_max_scaler::min_max_scaler(const real_type lower, const real_type upper) :
    min_max_scaler{ mpi::communicator{}, lower, upper } { }

min_max_scaler::min_max_scaler(mpi::communicator comm, const real_type lower, const real_type upper) :
    scaling_interval_{ std::make_pair(lower, upper) },
    comm_{ std::move(comm) } {
    if (lower >= upper) {
        throw min_max_scaler_exception{ fmt::format("Inconsistent scaling interval specification: lower ({}) must be less than upper ({})!", lower, upper) };
    }
}

min_max_scaler::min_max_scaler(const std::string &filename) :
    min_max_scaler{ mpi::communicator{}, filename } { }

min_max_scaler::min_max_scaler(mpi::communicator comm, const std::string &filename) :
    comm_{ std::move(comm) } {
    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // read scaling values from file
    std::tie(scaling_interval_, scaling_factors_) = detail::io::parse_scaling_factors<factors>(reader);
}

void min_max_scaler::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // write scaling values to file
    detail::io::write_scaling_factors(filename, scaling_interval_, scaling_factors_);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                comm_,
                "Write {} scaling factors in {} to the file '{}'.\n",
                detail::tracking::tracking_entry{ "scaling_factors_write", "num_scaling_factors", scaling_factors_.size() },
                detail::tracking::tracking_entry{ "scaling_factors_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "scaling_factors_write", "filename", filename });
}

}  // namespace plssvm

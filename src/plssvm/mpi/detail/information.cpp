/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/mpi/detail/information.hpp"

#include "plssvm/backend_types.hpp"                     // plssvm::backend_type
#include "plssvm/detail/logging/mpi_log_untracked.hpp"  // plssvm::detail::log_untracked
#include "plssvm/mpi/communicator.hpp"                  // plssvm::mpi::communicator
#include "plssvm/solver_types.hpp"                      // plssvm::solver_type
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                  // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <cstddef>   // std::size_t
#include <map>       // std::map
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::mpi::detail {

void gather_and_print_solver_information(const communicator &comm, solver_type rank_solver) {
    // gather the solver information from all MPI ranks on the main MPI rank
    const std::vector<solver_type> all_solvers = comm.gather(rank_solver);

    // output information only on the main MPI rank!
    if (comm.is_main_rank()) {
        // map all MPI ranks to its used solver
        std::map<solver_type, std::vector<std::size_t>> solvers_for_rank{};
        for (std::size_t i = 0; i < all_solvers.size(); ++i) {
            solvers_for_rank[all_solvers[i]].push_back(i);
        }

        // output the information (again, only on the main MPI rank)
        ::plssvm::detail::log_untracked(verbosity_level::full,
                                        comm,
                                        "\nThe used solver(s) for AX=B across {} MPI rank(s) are:\n",
                                        comm.size());
        for (const auto &[solver, ranks] : solvers_for_rank) {
            ::plssvm::detail::log_untracked(verbosity_level::full,
                                            comm,
                                            "  - {}: {}\n",
                                            solver,
                                            fmt::join(ranks, ", "));
        }
        ::plssvm::detail::log_untracked(verbosity_level::full,
                                        comm,
                                        "\n");
    }
}

void gather_and_print_csvm_information(const communicator &comm, backend_type rank_backend, target_platform rank_target, const std::vector<std::string> &rank_devices, const std::optional<std::string> &additional_info) {
    // gather the information from all MPI ranks on the main MPI rank
    const std::vector<backend_type> backends_per_ranks = comm.gather(rank_backend);
    const std::vector<target_platform> targets_per_rank = comm.gather(rank_target);
    // pre-process device names
    std::map<std::string, std::size_t> devices_for_rank{};
    for (const std::string &device : rank_devices) {
        ++devices_for_rank[device];
    }
    // assemble one device name string
    std::vector<std::string> rank_str{};
    rank_str.reserve(devices_for_rank.size());
    for (const auto &[device, count] : devices_for_rank) {
        rank_str.emplace_back(fmt::format("{}x {}", count, device));
    }
    const std::vector<std::string> strings_per_rank = comm.gather(fmt::format("{}", fmt::join(rank_str, ", ")));
    // get the potentially additional information
    const std::vector<std::string> additional_info_per_rank = comm.gather(additional_info.value_or(""));

    // output the information (again, only on the main MPI rank)
    ::plssvm::detail::log_untracked(verbosity_level::full,
                                    comm,
                                    "\nThe setup across {} MPI rank(s) is:\n",
                                    comm.size());
    for (std::size_t i = 0; i < comm.size(); ++i) {
        ::plssvm::detail::log_untracked(verbosity_level::full,
                                        comm,
                                        "  - {}: {}{} for {} ({})\n",
                                        i,
                                        backends_per_ranks[i],
                                        additional_info_per_rank[i].empty() ? "" : fmt::format(" ({})", additional_info_per_rank[i]),
                                        targets_per_rank[i],
                                        strings_per_rank[i]);
    }
}

void gather_and_print_csvm_information(const communicator &comm, backend_type rank_backend, target_platform rank_target, const std::optional<std::string> &additional_info) {
    // gather the information from all MPI ranks on the main MPI rank
    const std::vector<backend_type> backends_per_ranks = comm.gather(rank_backend);
    const std::vector<target_platform> targets_per_rank = comm.gather(rank_target);
    // get the potentially additional information
    const std::vector<std::string> additional_info_per_rank = comm.gather(additional_info.value_or(""));

    // output the information (again, only on the main MPI rank)
    ::plssvm::detail::log_untracked(verbosity_level::full,
                                    comm,
                                    "\nThe setup across {} MPI rank(s) is:\n",
                                    comm.size());
    for (std::size_t i = 0; i < comm.size(); ++i) {
        ::plssvm::detail::log_untracked(verbosity_level::full,
                                        comm,
                                        "  - {}: {}{} for {}\n",
                                        i,
                                        backends_per_ranks[i],
                                        additional_info_per_rank[i].empty() ? "" : fmt::format(" ({})", additional_info_per_rank[i]),
                                        targets_per_rank[i]);
    }
}

}  // namespace plssvm::mpi::detail

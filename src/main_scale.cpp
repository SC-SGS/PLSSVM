/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-scale` executable used for scaling a data set to a specified range.
 */

#include "plssvm/backend_types.hpp"                        // plssvm::backend_type
#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/data_set/data_set.hpp"                    // plssvm::optional_ref
#include "plssvm/detail/cmd/data_set_variants.hpp"         // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_scale.hpp"              // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/logging/mpi_log_untracked.hpp"     // plssvm::detail::log_untracked
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE,
#include "plssvm/detail/type_traits.hpp"                   // plssvm::detail::remove_cvref_t
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::exception, plssvm::cmd_parser_exit
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level
                                                           // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME
#include "plssvm/detail/utility.hpp"                       // PLSSVM_IS_DEFINED
#include "plssvm/environment.hpp"                          // plssvm::environment_scope_guard

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "hws/system_hardware_sampler.hpp"  // hws::system_hardware_sampler
#endif

#include "fmt/base.h"     // fmt::print
#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/format.h"   // fmt::format

#include <chrono>     // std::chrono::{steady_clock, duration}, std::chrono_literals namespace
#include <cstddef>    // std::size_t
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::endl
#include <memory>     // std::unique_ptr, std::make_unique
#include <variant>    // std::visit
#include <vector>     // std::vector

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
    // may throw an exception if the required level of MPI parallelism isn't available (really rare)
    std::unique_ptr<plssvm::environment::scope_guard> mpi_guard{};
    try {
        // initialize MPI environment only via the plssvm::scope_guard (by explicitly specifying NO backend)
        mpi_guard = std::make_unique<plssvm::environment::scope_guard>(std::vector<plssvm::backend_type>{});
    } catch (const plssvm::mpi_exception &e) {
        std::cerr << "An exception occurred while setting up MPI!: " << e.what_with_loc() << std::endl;
    }

    // create a PLSSVM communicator -> use MPI_COMM_WORLD for our executables
    // if MPI is not supported, does nothing
    const plssvm::mpi::communicator comm{};

    // plssvm-scale ONLY supports one MPI rank
    if (comm.size() > std::size_t{ 1 }) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format("Currently, plssvm-scale only supports a single MPI process, but {} where used!", comm.size()) << std::endl;
        }
        return EXIT_FAILURE;
    }

    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME(start_time);

        // create and start CPU hardware sampler if available
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
        hws::system_hardware_sampler sampler{ PLSSVM_HARDWARE_SAMPLING_INTERVAL };
        sampler.start_sampling();
#endif

        // create default parameters
        const plssvm::detail::cmd::parser_scale cmd_parser{ comm, argc, argv };

        // send warning if the build type is release and assertions are enabled
        if constexpr (std::string_view{ PLSSVM_BUILD_TYPE } == "Release" && PLSSVM_IS_DEFINED(PLSSVM_ENABLE_ASSERTS)) {
            plssvm::detail::log_untracked(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                          comm,
                                          "WARNING: The build type is set to Release, but assertions are enabled. "
                                          "This may result in a noticeable performance degradation in parts of PLSSVM!\n");
        }

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            comm,
                            "\ntask: scaling\n{}\n",
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // create data set and scale
        const auto data_set_visitor = [&](auto &&data) {
            // write scaled data to output file
            if (!cmd_parser.scaled_filename.empty()) {
                data.save(cmd_parser.scaled_filename, cmd_parser.format);
            } else {
                fmt::print("\n");
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type;

                // output to console if no output filename is provided
                const auto &matrix = data.data();
                const plssvm::optional_ref<const std::vector<label_type>> label = data.labels();
                for (std::size_t row = 0; row < matrix.num_rows(); ++row) {
                    if (label.has_value()) {
                        fmt::print(FMT_COMPILE("{} "), label.value().get()[row]);
                    }
                    for (std::size_t col = 0; col < matrix.num_cols(); ++col) {
                        if (matrix(row, col) != plssvm::real_type{ 0.0 }) {
                            fmt::print(FMT_COMPILE("{}:{:.10e} "), col + 1, matrix(row, col));
                        }
                    }
                    fmt::print("\n");
                }
            }

            // save scaling parameters if requested
            if (!cmd_parser.save_filename.empty() && data.scaling_factors().has_value()) {
                data.scaling_factors()->get().save(cmd_parser.save_filename);
            }
        };
        std::visit(data_set_visitor, plssvm::detail::cmd::data_set_factory(comm, cmd_parser));

        // stop CPU hardware sampler and dump results if available
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
        sampler.stop_sampling();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY(sampler);
#endif

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
                            comm,
                            "\nTotal runtime: {}\n",
                            plssvm::detail::tracking::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);

    } catch (const plssvm::cmd_parser_exit &e) {
        // something inside the cmd parser went wrong
        // -> don't call std::exit directly to gracefully tear down the environment
        return e.exit_code();
    } catch (const plssvm::exception &e) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        std::cerr << fmt::format("An exception occurred on MPI rank {}!: {}", comm.rank(), e.what_with_loc()) << std::endl;
#else
        std::cerr << "An exception occurred!: " << e.what_with_loc() << std::endl;
#endif
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        std::cerr << fmt::format("An exception occurred on MPI rank {}!: {}", comm.rank(), e.what()) << std::endl;
#else
        std::cerr << "An exception occurred!: " << e.what() << std::endl;
#endif
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-train` executable used for training a C-SVM model.
 */

#include "plssvm/core.hpp"
#include "plssvm/detail/cmd/data_set_variants.hpp"         // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_train.hpp"              // plssvm::detail::cmd::parser_train
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE,
                                                           // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"                       // PLSSVM_IS_DEFINED

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "hws/system_hardware_sampler.hpp"  // hws::system_hardware_sampler
#endif

#include "plssvm/mpi/detail/version.hpp"

#include "fmt/format.h"  // fmt::format

#include <algorithm>    // std::for_each
#include <chrono>       // std::chrono::{steady_clock, duration, milliseconds}, std::chrono_literals namespace
#include <cstddef>      // std::size_t
#include <cstdlib>      // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <functional>   // std::mem_fn
#include <iostream>     // std::cerr, std::endl
#include <memory>       // std::unique_ptr, std::make_unique
#include <string>       // std::string
#include <type_traits>  // std::remove_reference_t
#include <utility>      // std::pair
#include <variant>      // std::visit
#include <vector>       // std::vector

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
    // create environment scoped guard
    const plssvm::environment::scope_guard environment_guard{};
    // create a PLSSVM communicator -> use MPI_COMM_WORLD for our executables
    // if MPI is not supported, does nothing
    const plssvm::mpi::communicator comm{};

    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME(start_time);

        // create and start CPU hardware sampler if available
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
        hws::system_hardware_sampler sampler{ PLSSVM_HARDWARE_SAMPLING_INTERVAL };
        sampler.start_sampling();
#endif

        // parse SVM parameter from command line
        const plssvm::detail::cmd::parser_train cmd_parser{ comm, argc, argv };

        // add MPI related tracking entries
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "mpi", "", comm }));

        // send warning if the build type is release and assertions are enabled
        if constexpr (std::string_view{ PLSSVM_BUILD_TYPE } == "Release" && PLSSVM_IS_DEFINED(PLSSVM_ENABLE_ASSERTS)) {
            plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                comm,
                                "WARNING: The build type is set to Release, but assertions are enabled. "
                                "This may result in a noticeable performance degradation in parts of PLSSVM!\n");
        }

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            comm,
                            "\ntask: training\n{}\n\n\n",
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // create data set
        const auto data_set_visitor = [&](auto &&data) {
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // check whether SYCL is used as backend (it is either requested directly or as automatic backend)
            const bool use_sycl_as_backend{ cmd_parser.backend == plssvm::backend_type::sycl || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::sycl) };
            // check whether Kokkos is used as backend (it is either requested directly or as automatic backend)
            const bool use_kokkos_as_backend{ cmd_parser.backend == plssvm::backend_type::kokkos || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::kokkos) };

            // create SVM
            const std::unique_ptr<plssvm::csvm> svm = [&]() {
                if (use_sycl_as_backend) {
                    return plssvm::make_csvm(cmd_parser.backend, comm, cmd_parser.target, cmd_parser.csvm_params, plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type, plssvm::sycl_kernel_invocation_type = cmd_parser.sycl_kernel_invocation_type);
                } else if (use_kokkos_as_backend) {
                    return plssvm::make_csvm(cmd_parser.backend, comm, cmd_parser.target, cmd_parser.csvm_params, plssvm::kokkos_execution_space = cmd_parser.kokkos_execution_space);
                } else {
                    return plssvm::make_csvm(cmd_parser.backend, comm, cmd_parser.target, cmd_parser.csvm_params);
                }
            }();

            // only specify plssvm::max_iter if it isn't its default value
            const plssvm::model<label_type> model =
                cmd_parser.max_iter == std::size_t{ 0 }
                    ? svm->fit(data,
                               plssvm::epsilon = cmd_parser.epsilon,
                               plssvm::classification = cmd_parser.classification,
                               plssvm::solver = cmd_parser.solver)
                    : svm->fit(data,
                               plssvm::epsilon = cmd_parser.epsilon,
                               plssvm::max_iter = cmd_parser.max_iter,
                               plssvm::classification = cmd_parser.classification,
                               plssvm::solver = cmd_parser.solver);

            // save model to file
            model.save(cmd_parser.model_filename);
        };
        std::visit(data_set_visitor, plssvm::detail::cmd::data_set_factory(comm, cmd_parser));

        // stop CPU hardware sampler and dump results if available
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
        sampler.stop_sampling();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY(sampler);
#endif

        // wait until all MPI processes reach this point
        comm.barrier();

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full,
                            comm,
                            "\nTotal runtime: {}\n",
                            plssvm::detail::tracking::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        // TODO: really change file name? what to output on the command line?
        std::string performance_tracking_filename{ cmd_parser.performance_tracking_filename };
#if defined(PLSSVM_HAS_MPI_ENABLED)
        if (!performance_tracking_filename.empty()) {
            // only append rank name to the file name if a file name has been provided
            performance_tracking_filename += fmt::format(".{}", comm.rank());
        }
#endif
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(performance_tracking_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << fmt::format("An exception occurred on MPI rank {}!: {}", comm.rank(), e.what_with_loc()) << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << fmt::format("An exception occurred on MPI rank {}!: {}", comm.rank(), e.what()) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

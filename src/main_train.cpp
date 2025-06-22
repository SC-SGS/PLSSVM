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
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/logging/mpi_log_untracked.hpp"     // plssvm::detail::log_untracked
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE,
                                                           // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"                       // PLSSVM_IS_DEFINED
#include "plssvm/mpi/environment.hpp"                      // plssvm::mpi::is_executed_via_mpirun

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "hws/system_hardware_sampler.hpp"  // hws::system_hardware_sampler
#endif

#include "fmt/format.h"  // fmt::format

#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}, std::chrono_literals namespace
#include <cstddef>      // std::size_t
#include <cstdlib>      // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <filesystem>   // std::filesystem::path
#include <iostream>     // std::cerr, std::endl
#include <memory>       // std::unique_ptr, std::make_unique
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::remove_reference_t
#include <variant>      // std::visit
#include <vector>       // std::vector

using namespace std::chrono_literals;

/**
 * @brief Fit a C-SVC model using the provided C-SVC, classification data set and command line parser.
 * @tparam svm_type the type of the C-SVC
 * @tparam label_type the type of the labels
 * @param[in] svm the C-SVC used to fit the model
 * @param[in] data the classification data set used to train the model
 * @param[in] cmd_parser the command line parser containing the user provided parameters
 * @return the learned classification model (`[[nodiscard]]`)
 */
template <typename svm_type, typename label_type>
[[nodiscard]] plssvm::classification_model<label_type> fit_csvc(const svm_type &svm, const plssvm::classification_data_set<label_type> &data, const plssvm::detail::cmd::parser_train &cmd_parser) {
    if (cmd_parser.max_iter == std::size_t{ 0 }) {
        return svm.fit(data, plssvm::epsilon = cmd_parser.epsilon, plssvm::classification = cmd_parser.classification, plssvm::solver = cmd_parser.solver);
    } else {
        return svm.fit(data, plssvm::epsilon = cmd_parser.epsilon, plssvm::max_iter = cmd_parser.max_iter, plssvm::classification = cmd_parser.classification, plssvm::solver = cmd_parser.solver);
    }
}

/**
 * @brief Fit a C-SVR model using the provided C-SVR, regression data set and command line parser.
 * @tparam svm_type the type of the C-SVR
 * @tparam label_type the type of the labels
 * @param[in] svm the C-SVR used to fit the model
 * @param[in] data the regression data set used to train the model
 * @param[in] cmd_parser the command line parser containing the user provided parameters
 * @return the learned regression model (`[[nodiscard]]`)
 */
template <typename svm_type, typename label_type>
[[nodiscard]] plssvm::regression_model<label_type> fit_csvr(const svm_type &svm, const plssvm::regression_data_set<label_type> &data, const plssvm::detail::cmd::parser_train &cmd_parser) {
    if (cmd_parser.max_iter == std::size_t{ 0 }) {
        return svm.fit(data, plssvm::epsilon = cmd_parser.epsilon, plssvm::solver = cmd_parser.solver);
    } else {
        return svm.fit(data, plssvm::epsilon = cmd_parser.epsilon, plssvm::max_iter = cmd_parser.max_iter, plssvm::solver = cmd_parser.solver);
    }
}

int main(int argc, char *argv[]) {
    // initialize MPI environment only via the plssvm::scope_guard (by explicitly specifying NO backend)
    [[maybe_unused]] plssvm::environment::scope_guard mpi_guard{ {} };
    // create a PLSSVM communicator -> use MPI_COMM_WORLD for our executables
    // if MPI is not supported, does nothing
    plssvm::mpi::communicator comm{};

#if defined(PLSSVM_HAS_MPI_ENABLED)
    plssvm::detail::log_untracked(plssvm::verbosity_level::full,
                                  comm,
                                  "Using {} MPI rank(s) for our C-SVM.\n",
                                  comm.size());
#else
    if (plssvm::mpi::is_executed_via_mpirun()) {
        plssvm::detail::log_untracked(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                      comm,
                                      "WARNING: PLSSVM was built without MPI support, but plssvm-train was executed via mpirun! "
                                      "As a result, each MPI process will run the same code.\n");
    }
#endif

    // create std::unique_ptr containing a plssvm::scope_guard
    // -> used to automatically handle necessary environment teardown operations
    std::unique_ptr<plssvm::environment::scope_guard> environment_guard{};

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
            plssvm::detail::log_untracked(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                          comm,
                                          "WARNING: The build type is set to Release, but assertions are enabled. "
                                          "This may result in a noticeable performance degradation in parts of PLSSVM!\n");
        }

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            comm,
                            "\ntask: training ({})\n{}\n\n\n",
                            plssvm::svm_type_to_task_name(cmd_parser.svm),
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // update the load balancing weights if they were provided
        if (!cmd_parser.mpi_load_balancing_weights.empty()) {
            comm.set_load_balancing_weights(cmd_parser.mpi_load_balancing_weights);
        }

        // create data set
        const auto data_set_visitor = [&](auto &&data) {
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;
            using csvm_type = typename std::remove_reference_t<decltype(data)>::svm_fit_type;
            using model_type = typename csvm_type::template model_type<label_type>;

            // check whether SYCL is used as backend (it is either requested directly or as automatic backend)
            const bool use_sycl_as_backend{ cmd_parser.backend == plssvm::backend_type::sycl || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::sycl) };
            // check whether HPX is used as backend (it is either requested directly or as automatic backend)
            const bool use_hpx_as_backend{ cmd_parser.backend == plssvm::backend_type::hpx || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::hpx) };
            // check whether Kokkos is used as backend (it is either requested directly or as automatic backend)
            const bool use_kokkos_as_backend{ cmd_parser.backend == plssvm::backend_type::kokkos || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::kokkos) };

            // initialize environments if necessary
            std::vector<plssvm::backend_type> backends_to_initialize{};
            if (use_hpx_as_backend) {
                backends_to_initialize.push_back(plssvm::backend_type::hpx);
            }
            if (use_kokkos_as_backend) {
                backends_to_initialize.push_back(plssvm::backend_type::kokkos);
            }
            environment_guard = std::make_unique<plssvm::environment::scope_guard>(backends_to_initialize);

            // create SVM
            const std::unique_ptr<csvm_type> svm = [&]() {
                if (use_sycl_as_backend) {
                    return plssvm::make_csvm<csvm_type>(cmd_parser.backend, comm, cmd_parser.target, cmd_parser.csvm_params, plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type, plssvm::sycl_data_parallel_kernel = cmd_parser.sycl_data_parallel_kernel);
                } else if (use_kokkos_as_backend) {
                    return plssvm::make_csvm<csvm_type>(cmd_parser.backend, comm, cmd_parser.target, cmd_parser.csvm_params, plssvm::kokkos_execution_space = cmd_parser.kokkos_execution_space);
                } else {
                    return plssvm::make_csvm<csvm_type>(cmd_parser.backend, comm, cmd_parser.target, cmd_parser.csvm_params);
                }
            }();

            // only specify the named arguments available for the respective SVM type
            const model_type model = [&]() {
                if constexpr (std::is_same_v<csvm_type, plssvm::csvc>) {
                    return fit_csvc(*svm, data, cmd_parser);
                } else if constexpr (std::is_same_v<csvm_type, plssvm::csvr>) {
                    return fit_csvr(*svm, data, cmd_parser);
                } else {
                    // unreachable
                    plssvm::detail::unreachable();
                }
            }();
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

#if defined(PLSSVM_HAS_MPI_ENABLED)
        if (cmd_parser.performance_tracking_filename.empty()) {
            // be sure that the output tracking results are correctly serialized
            comm.serialize([&]() {
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);
            });
        } else {
            // update filename with MY MPI rank
            std::filesystem::path path{ cmd_parser.performance_tracking_filename };
            path.replace_filename(fmt::format("{}.{}{}", path.stem(), comm.rank(), path.extension()));
            // output to all files in parallel
            PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(path.string());
        }
#else
        // if not compiled with MPI, simply output the tracking information
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);
#endif

    } catch (const plssvm::cmd_parser_exit &e) {
        // something inside the cmd parser went wrong
        // -> don't call std::exit directly to gracefully tear down the environment
        return e.exit_code();
    } catch (const plssvm::exception &e) {
        std::cerr << fmt::format("An exception occurred on MPI rank {}!: {}", comm.rank(), e.what_with_loc()) << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << fmt::format("An exception occurred on MPI rank {}!: {}", comm.rank(), e.what()) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

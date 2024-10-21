/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-predict` executable used for predicting a data set using a previously computed C-SVM model.
 */

#include "plssvm/core.hpp"
#include "plssvm/detail/cmd/data_set_variants.hpp"         // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_predict.hpp"            // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE,
                                                           // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY
                                                           // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"                       // PLSSVM_IS_DEFINED

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    #include "Kokkos_Core.hpp"  // Kokkos::initialize, Kokkos::is_initialized, Kokkos::finalize, Kokkos::is_finalized
#endif

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "hws/system_hardware_sampler.hpp"  // hws::system_hardware_sampler
#endif

#include "fmt/format.h"  // fmt::print
#include "fmt/os.h"      // fmt::ostream, fmt::output_file
#include "fmt/ranges.h"  // fmt::join

#include <algorithm>   // std::for_each
#include <chrono>      // std::chrono::{steady_clock, duration}, std::chrono_literals namespace
#include <cstdlib>     // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>   // std::exception
#include <fstream>     // std::ofstream
#include <functional>  // std::mem_fn
#include <iostream>    // std::cerr, std::endl
#include <utility>     // std::pair
#include <variant>     // std::visit
#include <vector>      // std::vector

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME(start_time);

        // create and start CPU hardware sampler if available
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
        hws::system_hardware_sampler sampler{ PLSSVM_HARDWARE_SAMPLING_INTERVAL };
        sampler.start_sampling();
#endif

        // parse SVM parameter from command line
        const plssvm::detail::cmd::parser_predict cmd_parser{ argc, argv };

        // send warning if the build type is release and assertions are enabled
        if constexpr (std::string_view{ PLSSVM_BUILD_TYPE } == "Release" && PLSSVM_IS_DEFINED(PLSSVM_ENABLE_ASSERTS)) {
            plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                "WARNING: The build type is set to Release, but assertions are enabled. "
                                "This may result in a noticeable performance degradation in parts of PLSSVM!\n");
        }

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\ntask: prediction\n{}\n",
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // create data set
        const auto data_set_visitor = [&](auto &&data) {
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // check whether SYCL is used as backend (it is either requested directly or as automatic backend)
            const bool use_sycl_as_backend{ cmd_parser.backend == plssvm::backend_type::sycl || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::sycl) };

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
            // check whether Kokkos is used as backend (it is either requested directly or as automatic backend)
            const bool use_kokkos_as_backend{ cmd_parser.backend == plssvm::backend_type::kokkos || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::kokkos) };

            // initialize Kokkos if necessary
            if (use_kokkos_as_backend) {
                Kokkos::initialize(argc, argv);  // TODO: set device?
                PLSSVM_ASSERT(Kokkos::is_initialized(), "Something went wrong initializing the Kokkos environment!");
            }
#endif

            // create default csvm
            const std::unique_ptr<plssvm::csvm> svm = use_sycl_as_backend ? plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type)
                                                                          : plssvm::make_csvm(cmd_parser.backend, cmd_parser.target);

            // create model
            const plssvm::model<label_type> model{ cmd_parser.model_filename };

            // output parameter used to learn the model
            {
                const plssvm::parameter params = model.get_params();
                plssvm::detail::log(plssvm::verbosity_level::full,
                                    "Parameter used to train the model:\n"
                                    "  kernel_type: {} -> {}\n",
                                    params.kernel_type,
                                    plssvm::kernel_function_type_to_math_string(params.kernel_type));
                switch (params.kernel_type) {
                    case plssvm::kernel_function_type::linear:
                        break;
                    case plssvm::kernel_function_type::polynomial:
                        plssvm::detail::log(plssvm::verbosity_level::full,
                                            "  degree: {}\n"
                                            "  gamma: {}\n"
                                            "  coef0: {}\n",
                                            params.degree,
                                            plssvm::get_gamma_string(params.gamma),
                                            params.coef0);
                        break;
                    case plssvm::kernel_function_type::rbf:
                    case plssvm::kernel_function_type::laplacian:
                    case plssvm::kernel_function_type::chi_squared:
                        plssvm::detail::log(plssvm::verbosity_level::full, "  gamma: {}\n", plssvm::get_gamma_string(params.gamma));
                        break;
                    case plssvm::kernel_function_type::sigmoid:
                        plssvm::detail::log(plssvm::verbosity_level::full,
                                            "  gamma: {}\n"
                                            "  coef0: {}\n",
                                            plssvm::get_gamma_string(params.gamma),
                                            params.coef0);
                        break;
                }
            }

            // predict labels
            const std::vector<label_type> predicted_labels = svm->predict(model, data);

            // write prediction file
            {
                const std::chrono::time_point write_start_time = std::chrono::steady_clock::now();

                fmt::ostream out = fmt::output_file(cmd_parser.predict_filename);
                out.print("{}", fmt::join(predicted_labels, "\n"));

                const std::chrono::time_point write_end_time = std::chrono::steady_clock::now();
                plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
                                    "Write {} predictions in {} to the file '{}'.\n",
                                    plssvm::detail::tracking::tracking_entry{ "predictions_write", "num_predictions", predicted_labels.size() },
                                    plssvm::detail::tracking::tracking_entry{ "predictions_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(write_end_time - write_start_time) },
                                    plssvm::detail::tracking::tracking_entry{ "predictions_write", "filename", cmd_parser.predict_filename });
            }

            // print achieved accuracy (if possible)
            if (data.has_labels()) {
                // generate the classification report
                const std::vector<label_type> &correct_labels = *data.labels();
                const plssvm::classification_report report{ correct_labels, predicted_labels };

                // print complete report
                plssvm::detail::log(plssvm::verbosity_level::full, "\n{}\n", report);
                // print only accuracy for LIBSVM conformity
                plssvm::detail::log(plssvm::verbosity_level::libsvm, "{} (classification)\n", report.accuracy());
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "accuracy", "achieved_accuracy", report.accuracy().achieved_accuracy }));
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "accuracy", "num_correct", report.accuracy().num_correct }));
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "accuracy", "num_total", report.accuracy().num_total }));
            }

            // finalize Kokkos if necessary
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
            if (use_kokkos_as_backend) {  // TODO: what if an exception occurred?
                Kokkos::finalize();
                PLSSVM_ASSERT(Kokkos::is_finalized(), "Something went wrong finalizing the Kokkos environment!");
            }
#endif
        };
        std::visit(data_set_visitor, plssvm::detail::cmd::data_set_factory(cmd_parser));

        // stop CPU hardware sampler and dump results if available
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
        sampler.stop_sampling();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HWS_ENTRY(sampler);
#endif

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
                            "\nTotal runtime: {}\n",
                            plssvm::detail::tracking::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

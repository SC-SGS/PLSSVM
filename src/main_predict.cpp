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
#include "plssvm/detail/cmd/data_set_variants.hpp"              // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_predict.hpp"                 // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/logging.hpp"                            // plssvm::detail::log
#include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/hardware_sampler_factory.hpp"  // plssvm::detail::tracking::create_hardware_sampler
#include "plssvm/detail/tracking/performance_tracker.hpp"       // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/utility.hpp"                            // PLSSVM_IS_DEFINED

#include "fmt/format.h"  // fmt::print, fmt::join
#include "fmt/os.h"      // fmt::ostream, fmt::output_file

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

            // create default csvm
            const std::unique_ptr<plssvm::csvm> svm = use_sycl_as_backend ? plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type)
                                                                          : plssvm::make_csvm(cmd_parser.backend, cmd_parser.target);

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // initialize hardware sampling
            std::vector<std::unique_ptr<plssvm::detail::tracking::hardware_sampler>> sampler =
                plssvm::detail::tracking::create_hardware_sampler(svm->get_target_platform(), svm->num_available_devices());
            // start sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::start_sampling));
#endif

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

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // stop sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::stop_sampling));
            // write samples to yaml file
            std::for_each(sampler.cbegin(), sampler.cend(), [&](const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> &s) {
                using track_type = std::pair<plssvm::detail::tracking::hardware_sampler *, std::chrono::steady_clock::time_point>;
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "hardware_samples", s->device_identification(), track_type{ s.get(), start_time } }));
            });
#endif
        };
        std::visit(data_set_visitor, plssvm::detail::cmd::data_set_factory(cmd_parser));

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

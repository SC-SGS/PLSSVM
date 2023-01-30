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
#include "plssvm/detail/cmd/data_set_variants.hpp"
#include "plssvm/detail/cmd/parser_predict.hpp"
#include "plssvm/detail/logger.hpp"
#include "plssvm/detail/performance_tracker.hpp"

#include "fmt/chrono.h"   // directly print std::chrono literals with fmt
#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/format.h"   // fmt::format, fmt::print
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <chrono>         // std::chrono::{steady_clock, duration}
#include <cstdlib>        // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>      // std::exception
#include <fstream>        // std::ofstream
#include <iostream>       // std::cerr, std::clog, std::endl
#include <variant>        // std::visit
#include <vector>         // std::vector

int main(int argc, char *argv[]) {
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    try {
        // parse SVM parameter from command line
        plssvm::detail::cmd::parser_predict cmd_parser{ argc, argv };

        // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
        if (cmd_parser.backend != plssvm::backend_type::sycl && cmd_parser.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
            std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                     "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                                     cmd_parser.sycl_implementation_type)
                      << std::endl;
        }

        // output used parameter
        plssvm::detail::log("\ntask: prediction\n{}\n", plssvm::detail::tracking_entry{ "parameter", "", cmd_parser} );

        // create data set
        std::visit([&](auto &&data) {
            using real_type = typename std::remove_reference_t<decltype(data)>::real_type;
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // create model
            const plssvm::model<real_type, label_type> model{ cmd_parser.model_filename };
            // create default csvm
            auto svm = plssvm::make_csvm(cmd_parser.backend, cmd_parser.target);
            // predict labels
            const std::vector<label_type> predicted_labels = svm->predict(model, data);

            // write prediction file
            {
                const std::chrono::time_point start_time = std::chrono::steady_clock::now();

                fmt::ostream out = fmt::output_file(cmd_parser.predict_filename);
                out.print("{}", fmt::join(predicted_labels, "\n"));

                const std::chrono::time_point end_time = std::chrono::steady_clock::now();
                plssvm::detail::log("Write {} predictions in {} to the file '{}'.\n",
                                    plssvm::detail::tracking_entry{ "predictions_write", "num_predictions", predicted_labels.size() },
                                    plssvm::detail::tracking_entry{ "predictions_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                                    plssvm::detail::tracking_entry{ "predictions_write", "filename", cmd_parser.predict_filename });
            }

            // print achieved accuracy (if possible)
            if (data.has_labels()) {
                const std::vector<label_type> &correct_labels = data.labels().value();
                std::size_t correct{ 0 };
                for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
                    // check whether prediction is correct
                    if (predicted_labels[i] == correct_labels[i]) {
                        ++correct;
                    }
                }
                // print accuracy
                fmt::print("Accuracy = {}% ({}/{}) (classification)\n",
                           static_cast<real_type>(correct) / static_cast<real_type>(data.num_data_points()) * real_type{ 100 },
                           correct,
                           data.num_data_points());
            }
        },
                   plssvm::detail::cmd::data_set_factory(cmd_parser));
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    plssvm::detail::log("\nTotal runtime: {}\n", plssvm::detail::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    PLSSVM_PERFORMANCE_TRACKER_SAVE();
    return EXIT_SUCCESS;
}

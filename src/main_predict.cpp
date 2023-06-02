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

#include "plssvm/detail/cmd/data_set_variants.hpp"  // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_predict.hpp"     // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/logger.hpp"                 // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"    // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE

#include "fmt/format.h"                             // fmt::print, fmt::join
#include "fmt/os.h"                                 // fmt::ostream, fmt::output_file

#include <chrono>                                   // std::chrono::{steady_clock, duration}
#include <cstdlib>                                  // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>                                // std::exception
#include <fstream>                                  // std::ofstream
#include <iostream>                                 // std::cerr, std::clog, std::endl
#include <variant>                                  // std::visit
#include <vector>                                   // std::vector

int main(int argc, char *argv[]) {
    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

        // parse SVM parameter from command line
        const plssvm::detail::cmd::parser_predict cmd_parser{ argc, argv };

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\ntask: prediction\n{}\n",
                            plssvm::detail::tracking_entry{ "parameter", "", cmd_parser });

        // create data set
        std::visit([&](auto &&data) {
            using real_type = typename std::remove_reference_t<decltype(data)>::real_type;
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // create model
            const plssvm::model<real_type, label_type> model{ cmd_parser.model_filename };
            // create default csvm
            const auto svm = plssvm::make_csvm(cmd_parser.backend, cmd_parser.target);
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
                                    plssvm::detail::tracking_entry{ "predictions_write", "num_predictions", predicted_labels.size() },
                                    plssvm::detail::tracking_entry{ "predictions_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(write_end_time - write_start_time) },
                                    plssvm::detail::tracking_entry{ "predictions_write", "filename", cmd_parser.predict_filename });
            }

            // print achieved accuracy (if possible)
            if (data.has_labels()) {
                // generate the classification report
                const std::vector<label_type> &correct_labels = data.labels().value();
                const plssvm::classification_report report{ correct_labels, predicted_labels };

                // print complete report
                plssvm::detail::log(plssvm::verbosity_level::full, "\n{}\n", report);
                // print only accuracy for LIBSVM conformity
                plssvm::detail::log(plssvm::verbosity_level::libsvm, "{} (classification)\n", report.accuracy());
            }
        }, plssvm::detail::cmd::data_set_factory(cmd_parser));

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
                            "\nTotal runtime: {}\n",
                            plssvm::detail::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

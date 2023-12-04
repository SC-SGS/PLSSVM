/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-scale` executable used for scaling a data set to a specified range.
 */

#include "plssvm/core.hpp"

#include "plssvm/detail/cmd/data_set_variants.hpp"  // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_scale.hpp"       // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/logging.hpp"                // plssvm::detail::log
#include "plssvm/detail/performance_tracker.hpp"    // plssvm::detail::tracking_entry,PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE
#include "plssvm/matrix.hpp"                        // plssvm::matrix
#include "plssvm/verbosity_levels.hpp"              // plssvm::verbosity_level

#include <chrono>                                   // std::chrono::{steady_clock, duration}
#include <cstdlib>                                  // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>                                // std::exception
#include <iostream>                                 // std::cerr, std::endl
#include <utility>                                  // std::pair
#include <variant>                                  // std::visit

int main(int argc, char *argv[]) {
    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

        // create default parameters
        const plssvm::detail::cmd::parser_scale cmd_parser{ argc, argv };

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\ntask: scaling\n{}\n",
                            plssvm::detail::tracking_entry{ "parameter", "", cmd_parser });

        // create data set and scale
        std::visit([&](auto &&data) {
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

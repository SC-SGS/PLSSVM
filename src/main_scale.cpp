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
#include "plssvm/detail/cmd/data_set_variants.hpp"
#include "plssvm/detail/cmd/parser_scale.hpp"
#include "plssvm/detail/logger.hpp"
#include "plssvm/detail/performance_tracker.hpp"

#include <chrono>     // std::chrono::{steady_clock, duration}
#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <utility>    // std::pair
#include <variant>    // std::visit

int main(int argc, char *argv[]) {
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    try {
        // create default parameters
        plssvm::detail::cmd::parser_scale cmd_parser{ argc, argv };

        // output used parameter
        plssvm::detail::log("\ntask: scaling\n{}\n", plssvm::detail::tracking_entry{ "parameter", "", cmd_parser} );

        // create data set and scale
        std::visit([&](auto &&data) {
            // write scaled data to output file
            data.save(cmd_parser.scaled_filename, cmd_parser.format);

            // save scaling parameters if requested
            if (!cmd_parser.save_filename.empty() && data.scaling_factors().has_value()) {
                data.scaling_factors()->get().save(cmd_parser.save_filename);
            }
        }, plssvm::detail::cmd::data_set_factory(cmd_parser));
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    plssvm::detail::log("\nTotal runtime: {}\n", plssvm::detail::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    PLSSVM_PERFORMANCE_TRACKER_SAVE();
    return EXIT_SUCCESS;
}

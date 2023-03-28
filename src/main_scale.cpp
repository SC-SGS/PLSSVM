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

#include "fmt/core.h"  // fmt::print, fmt::format

#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <utility>    // std::pair
#include <variant>    // std::visit

int main(int argc, char *argv[]) {
    try {
        // create default parameters
        plssvm::detail::cmd::parser_scale cmd_parser{ argc, argv };

        // output used parameter
        if (plssvm::verbose) {
            fmt::print("\ntask: scaling\n{}\n", cmd_parser);
        }

        // create data set and scale
        std::visit([&](auto &&data) {
            // write scaled data to output file
            if (!cmd_parser.scaled_filename.empty()) {
                data.save(cmd_parser.scaled_filename, cmd_parser.format);
            } else {
                fmt::print("\n");
                using real_type = typename plssvm::detail::remove_cvref_t<decltype(data)>::real_type;
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type;

                // output to console if no output filename is provided
                const std::vector<std::vector<real_type>>& matrix = data.data();
                const plssvm::optional_ref<const std::vector<label_type>> label = data.labels();
                for (std::size_t row = 0; row < matrix.size(); ++row) {
                    if (label.has_value()) {
                        fmt::print(FMT_COMPILE("{} "), label.value().get()[row]);
                    }
                    for (std::size_t col = 0; col < matrix[row].size(); ++col) {
                        if (matrix[row][col] != real_type{ 0.0 }) {
                            fmt::print(FMT_COMPILE("{}:{:.10e} "), col + 1, matrix[row][col]);
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
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

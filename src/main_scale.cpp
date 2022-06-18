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

#include "fmt/core.h"     // fmt::print, fmt::format

#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <utility>    // std::pair
#include <variant>    // std::visit


int main(int argc, char *argv[]) {
    try {
        // create default parameters
        plssvm::detail::cmd::parameter_scale params{ argc, argv };

        // output used parameter
        if (plssvm::verbose) {
            fmt::print("\ntask: scaling\n{}\n", params);
        }

        // create data set and scale
        std::visit([&](auto&& data){
            // write scaled data to output file
            data.save(params.scaled_filename, params.format);

            // save scaling parameters if requested
            if (!params.save_filename.empty() && data.scaling_factors().has_value()) {
                data.scaling_factors()->get().save(params.save_filename);
            }

        }, plssvm::detail::data_set_factory(params));

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

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
#include <variant>    // std::variant, std::visit


// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

// two possible types: real_type + real_type and real_type + std::string
using data_set_variants = std::variant<plssvm::data_set<real_type>, plssvm::data_set<real_type, std::string>>;

// create variant based on runtime flag
data_set_variants data_set_factory(const plssvm::detail::cmd::parameter_scale<real_type>& params) {
    std::pair<real_type, real_type> scaling_bounds(params.lower, params.upper);
    if (params.base_params.strings_as_labels) {
        return data_set_variants{ plssvm::data_set<real_type, std::string>{ params.input_filename, std::move(scaling_bounds) } };
    } else {
        return data_set_variants{ plssvm::data_set<real_type>{ params.input_filename, std::move(scaling_bounds) } };
    }
}

int main(int argc, char *argv[]) {
    try {
        // create default parameters
        plssvm::detail::cmd::parameter_scale<real_type> params{ argc, argv };

        // output used parameter
        if (plssvm::verbose) {
            fmt::print("\ntask: scaling\n{}\n", params);
        }

        // create data set and scale
        std::visit([&](auto&& data){
            // write scaled data to output file
            data.save_data_set(params.scaled_filename, params.format);
        }, data_set_factory(params));

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

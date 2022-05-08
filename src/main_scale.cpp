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

#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <utility>    // std::pair

// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

int main(int argc, char *argv[]) {
    try {
        // parse parameter from command line
        plssvm::parameter_scale<real_type> params{ argc, argv };

        // output used parameter
        if (plssvm::verbose) {
            fmt::print("\n");
            fmt::print("task: scaling\n");
            fmt::print("lower: {}\n", params.lower);
            fmt::print("upper: {}\n", params.upper);
            fmt::print("input file (data set): '{}'\n", params.input_filename);
            fmt::print("output file (scaled data set): '{}'\n", params.scaled_filename);
            fmt::print("\n");
        }

        // create data set and scale
        plssvm::data_set<real_type> data{params.input_filename, std::pair<real_type, real_type>(params.lower, params.upper)};

        // write scaled data to output file
        data.save_data_set(params.scaled_filename, plssvm::file_format_type::libsvm);
        // TODO: arff?
        // TODO: label type from command line???

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

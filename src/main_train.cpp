/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-train` executable used for training a C-SVM model.
 */

#include "plssvm/core.hpp"
#include "plssvm/detail/cmd/data_set_variants.hpp"
#include "plssvm/detail/cmd/parser_train.hpp"
#include "plssvm/detail/logger.hpp"
#include "plssvm/detail/performance_tracker.hpp"

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <chrono>     // std::chrono::{steady_clock, duration}
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <variant>    // std::visit

int main(int argc, char *argv[]) {
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    try {
        // parse SVM parameter from command line
        plssvm::detail::cmd::parser_train cmd_parser{ argc, argv };

        // warn if kernel invocation type nd_range or hierarchical are explicitly set but SYCL isn't the current backend
        if (cmd_parser.backend != plssvm::backend_type::sycl && cmd_parser.sycl_kernel_invocation_type != plssvm::sycl::kernel_invocation_type::automatic) {
            std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                     "WARNING: explicitly set a SYCL kernel invocation type but the current backend isn't SYCL; ignoring --sycl_kernel_invocation_type={}",
                                     cmd_parser.sycl_kernel_invocation_type)
                      << std::endl;
        }
        // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
        if (cmd_parser.backend != plssvm::backend_type::sycl && cmd_parser.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
            std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                     "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                                     cmd_parser.sycl_implementation_type)
                      << std::endl;
        }

        // output used parameter
        plssvm::detail::log("\ntask: training\n{}\n\n\n", plssvm::detail::tracking_entry{ "parameter", "", cmd_parser} );

        // create data set
        std::visit([&](auto &&data) {
            using real_type = typename std::remove_reference_t<decltype(data)>::real_type;
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // create SVM
            std::unique_ptr<plssvm::csvm> svm;
            if (cmd_parser.backend == plssvm::backend_type::sycl) {
                svm = plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, cmd_parser.csvm_params,
                                        plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type,
                                        plssvm::sycl_kernel_invocation_type = cmd_parser.sycl_kernel_invocation_type);
            } else {
                svm = plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, cmd_parser.csvm_params);
            }
            // learn model
            if (cmd_parser.max_iter.is_default()) {
                cmd_parser.max_iter = data.num_data_points();
            }
            const plssvm::model<real_type, label_type> model = svm->fit(data, plssvm::epsilon = cmd_parser.epsilon, plssvm::max_iter = cmd_parser.max_iter);
            // save model to file
            model.save(cmd_parser.model_filename);
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

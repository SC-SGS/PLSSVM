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

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <variant>    // std::visit


int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::detail::cmd::parameter_train params{ argc, argv };

        std::visit([&](auto&& args) {
            // warn if kernel invocation type nd_range or hierarchical are explicitly set but SYCL isn't the current backend
            if (args.backend != plssvm::backend_type::sycl && args.sycl_kernel_invocation_type != plssvm::sycl::kernel_invocation_type::automatic) {
                std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                         "WARNING: explicitly set a SYCL kernel invocation type but the current backend isn't SYCL; ignoring --sycl_kernel_invocation_type={}",
                                         args.sycl_kernel_invocation_type)
                          << std::endl;
            }
            // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
            if (args.backend != plssvm::backend_type::sycl && args.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
                std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                         "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                                         args.sycl_implementation_type)
                          << std::endl;
            }

            // output used parameter
            if (plssvm::verbose) {
                fmt::print("\ntask: training\n{}\n", params);
            }

            // create data set
            std::visit([&](auto&& data){
                // TODO: put code here
            }, plssvm::detail::data_set_factory<false>(params));

        }, params.base_params);



        // create data set
//        plssvm::data_set<real_type> data{ params.input_filename };

        // create SVM
//        auto svm = plssvm::make_csvm(params);

        // learn
//        svm->learn();

        // save model file
//        svm->write_model(params.model_filename);

        std::vector<std::vector<double>> matrix = { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
//        for (const auto& row : matrix) {
//            for (const auto& col : row) {
//                std::cout << col << ' ';
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;

        std::vector<std::string> label{ "cat", "dog", "cat" };
        plssvm::data_set<double, std::string> data{ std::move(matrix), std::move(label), std::pair<double, double>(-1, 1)};
//        data.scale(-2, +2);

        for (const auto& row : data.data()) {
            for (const auto& col : row) {
                std::cout << col << ' ';
            }
            std::cout << std::endl;
        }

        auto lab = data.mapped_labels().value().get();
        for (const auto& l : lab) {
            std::cout << l << ' ';
        }
        std::cout << std::endl;
        auto la = data.labels().value().get();
        for (const auto& l : la) {
            std::cout << l << ' ';
        }
        std::cout << std::endl;
//
//        data.save_data_set("data.libsvm", plssvm::file_format_type::libsvm);
////        data.save_data_set("data.arff", plssvm::file_format_type::arff);
//
//        plssvm::data_set<double> data2{"../data/data.libsvm"};
////        for (const auto& row : data2.data()) {
////            for (const auto& col : row) {
////                std::cout << col << ' ';
////            }
////            std::cout << std::endl;
////        }

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

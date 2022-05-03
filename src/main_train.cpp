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

#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl

// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::parameter_train<real_type> params{ argc, argv };

        // warn if kernel invocation type nd_range or hierarchical are explicitly set but SYCL isn't the current backend
        if (params.backend != plssvm::backend_type::sycl && params.sycl_kernel_invocation_type != plssvm::sycl::kernel_invocation_type::automatic) {
            std::clog << fmt::format(
                "WARNING: explicitly set a SYCL kernel invocation type but the current backend isn't SYCL; ignoring --sycl_kernel_invocation_type={}",
                params.sycl_kernel_invocation_type)
                      << std::endl;
        }
        // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
        if (params.backend != plssvm::backend_type::sycl && params.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
            std::clog << fmt::format(
                "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                params.sycl_implementation_type)
                      << std::endl;
        }

        // output used parameter
        if (plssvm::verbose) {
            fmt::print("\n");
            fmt::print("task: training\n");
            fmt::print("kernel type: {} -> ", params.kernel);
            switch (params.kernel) {
                case plssvm::kernel_type::linear:
                    fmt::print("u'*v\n");
                    break;
                case plssvm::kernel_type::polynomial:
                    fmt::print("(gamma*u'*v + coef0)^degree\n");
                    fmt::print("gamma: {}\n", params.gamma);
                    fmt::print("coef0: {}\n", params.coef0);
                    fmt::print("degree: {}\n", params.degree);
                    break;
                case plssvm::kernel_type::rbf:
                    fmt::print("exp(-gamma*|u-v|^2)\n");
                    fmt::print("gamma: {}\n", params.gamma);
                    break;
            }
            fmt::print("cost: {}\n", params.cost);
            fmt::print("epsilon: {}\n", params.epsilon);
            fmt::print("input file (data set): '{}'\n", params.input_filename);
            fmt::print("output file (model): '{}'\n", params.model_filename);
            fmt::print("\n");
        }

        // create data set
        plssvm::data_set<real_type> data{ params.input_filename };

        // create SVM
        auto svm = plssvm::make_csvm(params);

        // learn
        svm->learn();

        // save model file
        svm->write_model(params.model_filename);

//        std::vector<std::vector<double>> matrix = { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };
//        for (const auto& row : matrix) {
//            for (const auto& col : row) {
//                std::cout << col << ' ';
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//
//        plssvm::data_set<double> data{ std::move(matrix) };
//        data.scale(-2, +2);
//
//        for (const auto& row : data.data()) {
//            for (const auto& col : row) {
//                std::cout << col << ' ';
//            }
//            std::cout << std::endl;
//        }
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

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

#include "fmt/chrono.h"   // directly print std::chrono literals with fmt
#include "fmt/format.h"   // fmt::format, fmt::print
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <chrono>     // std::chrono
#include <exception>  // std::exception
#include <fstream>    // std::ofstream
#include <iostream>   // std::cerr, std::clog, std::endl
#include <vector>     // std::vector

// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::parameter_predict<real_type> params{ argc, argv };

        // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
        if (params.backend != plssvm::backend_type::sycl && params.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
            std::clog << fmt::format(
                "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                params.sycl_implementation_type)
                      << std::endl;
        }

        // output used parameter
        if (params.print_info) {
            fmt::print("\n");
            fmt::print("task: prediction\n");
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
            fmt::print("rho: {}\n", params.rho);
            fmt::print("input file (data set): '{}'\n", params.input_filename);
            fmt::print("input file (model): '{}'\n", params.model_filename);
            fmt::print("output file (prediction): '{}'\n", params.predict_filename);
            fmt::print("\n");
        }

        // create SVM
        auto svm = plssvm::make_csvm(params);

        // predict labels
        const std::vector<real_type> labels = svm->predict_label(*params.test_data_ptr);

        // write prediction file
        {
            auto start_time = std::chrono::steady_clock::now();
            std::ofstream out{ params.predict_filename };
            out << fmt::format("{}", fmt::join(labels, "\n"));
            auto end_time = std::chrono::steady_clock::now();
            if (params.print_info) {
                fmt::print("Wrote prediction file ('{}') with {} labels in {}.\n",
                           params.predict_filename,
                           labels.size(),
                           std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
            }
        }

        // print achieved accuracy if possible
        if (params.value_ptr) {
            unsigned long long correct = 0;
            for (typename std::vector<real_type>::size_type i = 0; i < labels.size(); ++i) {
                // check of prediction was correct
                if ((*params.value_ptr)[i] * labels[i] > real_type{ 0.0 }) {
                    ++correct;
                }
            }
            // print accuracy
            fmt::print("Accuracy = {}% ({}/{}) (classification)\n",
                       static_cast<real_type>(correct) / static_cast<real_type>(params.test_data_ptr->size()) * real_type{ 100 },
                       correct,
                       params.test_data_ptr->size());
        }

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

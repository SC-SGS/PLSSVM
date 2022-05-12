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
#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/format.h"   // fmt::format, fmt::print
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <chrono>     // std::chrono
#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <fstream>    // std::ofstream
#include <iostream>   // std::cerr, std::clog, std::endl
#include <variant>    // std::variant, std::visit
#include <vector>     // std::vector


// perform calculations in single precision if requested
#ifdef PLSSVM_EXECUTABLES_USE_SINGLE_PRECISION
using real_type = float;
#else
using real_type = double;
#endif

// two possible types: real_type + real_type and real_type + std::string
using data_set_variants = std::variant<plssvm::data_set<float>, plssvm::data_set<float, std::string>, plssvm::data_set<double>, plssvm::data_set<double, std::string>>;

// create variant based on runtime flag
data_set_variants data_set_factory(const plssvm::detail::cmd::parameter_predict& params) {
    bool use_float;
    bool use_strings;
    std::visit([&](auto&& args) {
        use_float = args.float_as_real_type;
        use_strings = args.strings_as_labels;
    }, params.base_params);

    if (use_float && use_strings) {
        return data_set_variants{ plssvm::data_set<float, std::string>{ params.input_filename } };
    } else if (use_float && !use_strings) {
        return data_set_variants{ plssvm::data_set<float>{ params.input_filename } };
    } else if (!use_float && use_strings) {
        return data_set_variants{ plssvm::data_set<double, std::string>{ params.input_filename } };
    } else {
        return data_set_variants{ plssvm::data_set<double>{ params.input_filename } };
    }
}

int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::detail::cmd::parameter_predict params{ argc, argv };

        std::visit([&](auto&& args) {
            // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
            if (args.backend != plssvm::backend_type::sycl && args.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
                std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                         "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                                         args.sycl_implementation_type)
                          << std::endl;
            }

            // output used parameter
            if (plssvm::verbose) {
                fmt::print("\ntask: prediction\n{}\n", params);
            }

            // create data set
            std::visit([&](auto &&data) {
                // TODO: put code here
            }, data_set_factory(params));

            //        // create SVM
            //        auto svm = plssvm::make_csvm(params.base_params);
            //
            //        // predict labels
            //        const std::vector<real_type> labels = svm->predict_label(*params.test_data_ptr);
            //
            //        // write prediction file
            //        {
            //            auto start_time = std::chrono::steady_clock::now();
            //            std::ofstream out{ params.predict_filename };
            //            out << fmt::format("{}", fmt::join(labels, "\n"));
            //            auto end_time = std::chrono::steady_clock::now();
            //            if (params.print_info) {
            //                fmt::print("Wrote prediction file ('{}') with {} labels in {}.\n",
            //                           params.predict_filename,
            //                           labels.size(),
            //                           std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
            //            }
            //        }
            //
            //        // print achieved accuracy if possible
            //        if (params.value_ptr) {
            //            unsigned long long correct = 0;
            //            for (typename std::vector<real_type>::size_type i = 0; i < labels.size(); ++i) {
            //                // check of prediction was correct
            //                if ((*params.value_ptr)[i] * labels[i] > real_type{ 0.0 }) {
            //                    ++correct;
            //                }
            //            }
            //            // print accuracy
            //            fmt::print("Accuracy = {}% ({}/{}) (classification)\n",
            //                       static_cast<real_type>(correct) / static_cast<real_type>(params.test_data_ptr->size()) * real_type{ 100 },
            //                       correct,
            //                       params.test_data_ptr->size());
            //        }
        }, params.base_params);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

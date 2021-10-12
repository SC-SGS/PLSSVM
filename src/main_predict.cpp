/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `svm-predict` executable used for predicting a data set using a previously computed C-SVM model.
 */

#include "plssvm/core.hpp"

#include "fmt/format.h"  // fmt::format, fmt::print

#include <cstddef>    // std::size_t
#include <exception>  // std::exception
#include <fstream>    // std::ofstream
#include <iostream>   // std::cerr, std::endl
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

        // create SVM
        auto svm = plssvm::make_csvm(params);

        // predict labels
        std::vector<real_type> labels = svm->predict_label(*params.test_data_ptr);

        // write prediction file
        {
            std::ofstream out{ params.predict_filename };
            out << fmt::format("{}", fmt::join(labels, "\n"));
        }

        // print achieved accuracy if possible
        if (params.value_ptr) {
            std::size_t correct = 0;
            for (std::size_t i = 0; i < labels.size(); ++i) {
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

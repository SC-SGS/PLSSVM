/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief TODO: brief description
 */

#include "plssvm/core.hpp"

#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::endl

int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::parameter_train<double> params{ argc, argv };

        // create SVM
        auto svm = plssvm::make_csvm(params);

        // learn
        svm->learn();

        // save model file
        svm->write_model(params.model_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

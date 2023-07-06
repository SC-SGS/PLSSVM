/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::model class.
 */

#include "plssvm/core.hpp"

int main() {
    // create a data set from a file
    const plssvm::data_set data{ "path/to/train/file.libsvm" };

    // create a support vector machine
    const auto svm = plssvm::make_csvm();

    // fit the support vector machine
    const plssvm::model model = svm->fit(data);

    // save the model file
    model.save("path/to/model.libsvm");

    return 0;
}
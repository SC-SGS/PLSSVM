/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::regression_model class.
 */

#include "plssvm/core.hpp"

int main() {
    // create a regression data set from a file
    const plssvm::regression_data_set data{ "path/to/train/file.libsvm" };

    // create a support vector machine for the regression task
    const auto svr = plssvm::make_csvr();

    // fit the support vector machine
    const plssvm::regression_model model = svr->fit(data);

    // save the model file
    model.save("path/to/model.libsvm");

    return 0;
}

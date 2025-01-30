/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::csvr classes.
 */

#include "plssvm/core.hpp"

#include <string>
#include <vector>

int main() {
    // create a train regression data set from a file with int labels
    const plssvm::regression_data_set train_data_with_label{ "path/to/train/file.libsvm" };
    // create a test regression data set from a file without labels
    const plssvm::regression_data_set test_data{ "path/to/test/file.libsvm" };

    // create a support vector machine
    auto svr = plssvm::make_csvr();

    // optional: update a parameter; can also be directly passed to the plssvm::make_csvr function!
    svr->set_params(plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.001);

    // fit the support vector machine
    const plssvm::regression_model model = svr->fit(train_data_with_label);

    // score a new, unseen data set
    const double score = svr->score(model, test_data);

    //
    // Note: the model is NOT bound to a specific support vector machine
    //
    // explicitly make an OpenCL support vector machine
    const plssvm::opencl::csvr opencl_svr{};

    // predict labels
    const std::vector<int> predicted_labels = opencl_svr.predict(model, test_data);

    return 0;
}

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::csvm classes.
 */

#include "plssvm/core.hpp"

#include <string>
#include <vector>

int main() {
    // create a train data set from a file with int labels
    const plssvm::data_set train_data_with_label{ "path/to/train/file.libsvm" };
    // create a test data set from a file without labels
    const plssvm::data_set test_data{ "path/to/test/file.libsvm" };

    // create a support vector machine
    auto svm = plssvm::make_csvm();

    // optional: update a parameter; can also be directly passed to the plssvm::make_csvm function!
    svm->set_params(plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.001);

    // fit the support vector machine
    const plssvm::model model = svm->fit(train_data_with_label);

    // score the learned model
    const double model_score = svm->score(model);

    // score a new, unseen data set
    const double score = svm->score(model, test_data);

    //
    // Note: the model is NOT bound to a specific support vector machine
    //
    // explicitly make an OpenCL support vector machine
    const plssvm::opencl::csvm opencl_svm{};

    // predict labels
    const std::vector<int> predicted_labels = opencl_svm.predict(model, test_data);

    return 0;
}
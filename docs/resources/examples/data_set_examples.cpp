/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::data_set class.
 */

#include "plssvm/core.hpp"

#include <string>
#include <vector>

int main() {

    // create a data set from a file with int labels
    plssvm::data_set data1{ "path/to/train/file.libsvm" };
    // create a data set from a file with std::string labels
    plssvm::data_set std::string> data2{ "path/to/train/file_string_labels.libsvm" };

    // create a data set from a std::vector with labels
    std::vector<std::vector<double>> data_vector{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
    std::vector<int> label_vector{ 0, 0, 1 };
    plssvm::data_set data3{ data_vector, std::move(label_vector) };
    // get the different labels
    const bool has_label = data3.has_labels();        // will return true, since labels have explicitly been provided
    auto different_labels = data3.classes().value();  // will return an optional vector containing { 0, 1 }

    // create a train data set and scale it to the range [-1, 1]
    plssvm::data_set train_data{ "path/to/train/file.libsvm", { -1.0, 1.0 } };
    // scale a test data set and scale it according to the train data set
    plssvm::data_set test_data{ "path/to/test/file.libsvm", train_data.scaling_factors()->get() };

    return 0;
}
#include "plssvm/core.hpp"

#include <exception>
#include <iostream>
#include <vector>

int main() {
    try {
        // create a new C-SVM parameter set, explicitly overriding the default kernel function
        const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial };

        // create two data sets: one with the training data scaled to [-1, 1]
        // and one with the test data scaled like the training data
        const plssvm::data_set train_data{ "train_file.libsvm", { -1.0, 1.0 } };
        const plssvm::data_set test_data{ "test_file.libsvm", train_data.scaling_factors()->get() };

        // create C-SVM using the default backend and the previously defined parameter
        const auto svm = plssvm::make_csvm(params);

        // fit using the training data, (optionally) set the termination criterion
        const plssvm::model model = svm->fit(train_data, plssvm::epsilon = 10e-6);

        // get accuracy of the trained model
        const double model_accuracy = svm->score(model);
        std::cout << "model accuracy: " << model_accuracy << std::endl;

        // predict the labels
        const std::vector<int> predicted_label = svm->predict(model, test_data);
        // output a more complete classification report
        const std::vector<int> &correct_label = test_data.labels().value();
        std::cout << plssvm::classification_report{ correct_label, predicted_label } << std::endl;

        // write model file to disk
        model.save("model_file.libsvm");

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
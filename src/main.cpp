#include <plssvm/core.hpp>

#include <exception>
#include <iostream>

int main(int argc, char *argv[]) {
    // parse SVM parameter from command line
    plssvm::parameter<double> params{ argc, argv };

    try {
        // create SVM
        auto svm = plssvm::make_SVM(params);

        // learn
        svm->learn(params.input_filename, params.model_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

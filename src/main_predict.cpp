#include "plssvm/core.hpp"

#include <exception>  // std::exception
#include <iostream>   // std::clog, std::cerr, std::endl

int main(int argc, char *argv[]) {
    // parse SVM parameter from command line
    plssvm::parameter_predict<double> params{ argc, argv };
    //    std::clog << params << std::endl;

    try {
        // create SVM
        auto svm = plssvm::make_csvm(params);

        int correct = 0;
        for (int i = 0; i < params.test_data_ptr->size(); ++i) {
            double label = svm->predict((*(params.test_data_ptr))[i]);
            if (params.value_ptr && (*(params.value_ptr))[i] * label > 0.0) {
                ++correct;
            }
        }

        if (params.value_ptr) {
            std::cout << "accuracy: " << static_cast<double>(correct) / params.test_data_ptr->size() << std::endl;
        }

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

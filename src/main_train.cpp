/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/core.hpp"

#include <exception>  // std::exception
#include <iostream>   // std::clog, std::cerr, std::endl

int main(int argc, char *argv[]) {
    // parse SVM parameter from command line
    plssvm::parameter_train<double> params{ argc, argv };
    //    std::clog << params << std::endl;

    try {
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

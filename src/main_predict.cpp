#include "plssvm/core.hpp"

#include <exception>  // std::exception
#include <fstream>    // ofstream
#include <iostream>   // std::clog, std::cerr, std::endl

int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::parameter_predict<double> params{ argc, argv };
        // create SVM
        auto svm = plssvm::make_csvm(params);
        std::ofstream out(params.predict_filename);
        std::cout << out.fail() << out.is_open() << params.predict_filename << std::endl;
        int correct = 0;
        for (std::size_t i = 0; i < params.test_data_ptr->size(); ++i) {
            double label = svm->predict((*(params.test_data_ptr))[i]);
            out << label << '\n';
            if (params.value_ptr && (*(params.value_ptr))[i] * label > 0.0) {
                ++correct;
            }
        }
        out.close();
        if (params.value_ptr) {
            std::cout << "accuracy: " << static_cast<double>(correct) / params.test_data_ptr->size() * 100 << "%" << std::endl;
        }

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

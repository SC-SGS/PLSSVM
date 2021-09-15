/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright
*/

#include "plssvm/core.hpp"

#include <exception>  // std::exception
#include <fstream>    // std::ofstream
#include <iostream>   // std::clog, std::cerr, std::endl

int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::parameter_predict<double> params{ argc, argv };

        using real_type = typename decltype(params)::real_type;
        using size_type = typename decltype(params)::size_type;

        // create SVM
        auto svm = plssvm::make_csvm(params);

        std::ofstream out{ params.predict_filename };
        size_type correct = 0;
        for (size_type i = 0; i < params.test_data_ptr->size(); ++i) {
            // predict data point
            const real_type label = svm->predict((*params.test_data_ptr)[i]);
            // write label to file
            out << label << '\n';
            // check of prediction was correct
            if (params.value_ptr && (*params.value_ptr)[i] * label > real_type{ 0.0 }) {
                ++correct;
            }
        }

        // print achieved accuracy if possible
        if (params.value_ptr) {
            std::cout << "accuracy: " << static_cast<real_type>(correct) / static_cast<real_type>(params.test_data_ptr->size()) * 100 << " %" << std::endl;
        }

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

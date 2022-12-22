/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::make_csvm function.
 */

#include "plssvm/core.hpp"

int main() {
    // create a default support vector machine
    // the used backend is determined by the plssvm::determine_default_backend function
    const auto svm1 = plssvm::make_csvm();

    // explicitly define a backend to use (using the default SVM parameters)
    const auto svm2 = plssvm::make_csvm(plssvm::backend_type::cuda);

    // for SYCL, the SYCL implementation type can/must also be specified
    const auto svm3 = plssvm::make_csvm(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::dpcpp);

    // explicitly define a backend and parameters to use
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.0001 };
    const auto svm4 = plssvm::make_csvm(plssvm::backend_type::openmp, params);

    // explicitly define a backend and named-parameters to use
    const auto svm5 = plssvm::make_csvm(plssvm::backend_type::opencl, plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.0001);

    // explicitly define a backend, parameters, and named-parameters to use
    const auto svm6 = plssvm::make_csvm(plssvm::backend_type::sycl, params, plssvm::degree = 6);
    // Note: in this case the plssvm::parameter object must be THE FIRST parameter, i.e., the following will not compiled
    // const auto svm7 = plssvm::make_csvm(plssvm::backend_type::sycl, plssvm::degree = 6, params);

    return 0;
}
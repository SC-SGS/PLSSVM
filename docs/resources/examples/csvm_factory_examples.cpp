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
    // create a default classification support vector machine
    // the used backend is determined by the plssvm::determine_default_backend function
    const auto svc1 = plssvm::make_csvc();

    // explicitly define a backend to use (using the default SVM parameters)
    const auto svc2 = plssvm::make_csvc(plssvm::backend_type::cuda);

    // for SYCL, the SYCL implementation type can/must also be specified
    const auto svc3 = plssvm::make_csvc(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::dpcpp);

    // explicitly define a backend and parameters to use
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.0001 };
    const auto svc4 = plssvm::make_csvc(plssvm::backend_type::openmp, params);

    // explicitly define a backend and named-parameters to use
    const auto svc5 = plssvm::make_csvc(plssvm::backend_type::opencl, plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.0001);

    // explicitly define a backend, parameters, and named-parameters to use
    const auto svc6 = plssvm::make_csvc(plssvm::backend_type::sycl, params, plssvm::degree = 6);
    // Note: in this case the plssvm::parameter object must be THE FIRST parameter, i.e., the following will not compiled
    // const auto svc7 = plssvm::make_csvm(plssvm::backend_type::sycl, plssvm::degree = 6, params);

    // the same can be done for a regression support vector machine
    const auto svr1 = plssvm::make_csvr();
    const auto svr2 = plssvm::make_csvr(plssvm::backend_type::cuda);
    // etc.

    // additionally, the respective function can be determined with a non-type template parameter
    const auto svm1 = plssvm::make_csvm<plssvm::csvc>();
    const auto svm2 = plssvm::make_csvm<plssvm::csvr>(plssvm::backend_type::cuda);
    // etc.

    return 0;
}

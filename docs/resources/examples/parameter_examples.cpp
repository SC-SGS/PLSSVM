/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A few examples for the plssvm::parameter class.
 */

#include "plssvm/core.hpp"

int main() {
    // create default parameter
    const plssvm::parameter params1{};

    // create parameters by explicitly specifying ALL values (not recommended)
    const plssvm::parameter params2{
        plssvm::kernel_function_type::polynomial,  // kernel function
        4,                                         // degree
        0.001,                                     // gamma
        1.2,                                       // coef0
        0.01                                       // cost
    };

    // create parameters by using name-parameters
    const plssvm::parameter params3{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial, plssvm::degree = 2, plssvm::gamma = 0.001 };

    // create parameters by using another plssvm::parameter object together with named-parameters
    // initializes params4 with the values of params2 and OVERRIDES all values provided using named-parameters
    const plssvm::parameter params4{ params2, plssvm::degree = 5 };
    // Note: in this case the plssvm::parameter object must be THE FIRST parameter, i.e., the following will not compiled
    // const plssvm::parameter params5{ plssvm::degree = 5, params2 };

    return 0;
}
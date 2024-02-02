/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "pybind11/pybind11.h"  // py::module_, py::enum_

namespace py = pybind11;

void init_kernel_function_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::kernel_function_type>(m, "KernelFunctionType")
        .value("LINEAR", plssvm::kernel_function_type::linear, "linear kernel function: <u, v>")
        .value("POLYNOMIAL", plssvm::kernel_function_type::polynomial, "polynomial kernel function: (gamma * <u, v> + coef0)^degree")
        .value("RBF", plssvm::kernel_function_type::rbf, "radial basis function: e^(-gamma * ||u - v||^2)");

    // bind free functions
    m.def("kernel_function_type_to_math_string", &plssvm::kernel_function_type_to_math_string, "return the mathematical representation of a KernelFunctionType");
}

#include "plssvm/kernel_function_types.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_kernel_function_types(py::module &m) {
    // bind enum class
    py::enum_<plssvm::kernel_function_type>(m, "kernel_function_type")
        .value("linear", plssvm::kernel_function_type::linear)
        .value("polynomial", plssvm::kernel_function_type::polynomial)
        .value("rbf", plssvm::kernel_function_type::rbf);

    // bind free functions
    m.def("kernel_function_type_to_math_string", &plssvm::kernel_function_type_to_math_string);
}
#include "pybind11/pybind11.h"  // PYBIND11_MODULE

namespace py = pybind11;

// forward declare binding functions
void init_target_platforms(py::module &);
void init_backend_types(py::module &);
void init_file_format_types(py::module &);
void init_kernel_function_types(py::module &);
void init_parameter(py::module &);

PYBIND11_MODULE(plssvm, m) {
    // NOTE: the order matters. DON'T CHANGE IT!
    init_target_platforms(m);
    init_backend_types(m);
    init_file_format_types(m);
    init_kernel_function_types(m);
    init_parameter(m);
}
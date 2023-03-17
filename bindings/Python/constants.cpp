#include "plssvm/constants.hpp"

#include "pybind11/pybind11.h"  // py::module_

namespace py = pybind11;

void init_constants(py::module_ &m) {
    // enable or disable verbose output
    m.def("quiet", [](const bool verb) { plssvm::verbose = !verb; });
}
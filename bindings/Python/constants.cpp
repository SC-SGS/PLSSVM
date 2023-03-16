#include "plssvm/constants.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

// dummy class
class verbose {};

void init_constants(py::module &m) {
    // TODO: other name? (plssvm.verbose.verbose looks ugly)
    py::class_<verbose>(m, "verbose")
        .def_property_static(
            "verbose",
            [](py::object) { return plssvm::verbose; },
            [](py::object, const bool verb) { plssvm::verbose = verb; });
}
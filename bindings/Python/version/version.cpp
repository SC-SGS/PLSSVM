#include "plssvm/version/version.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::object
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

// dummy class
class version {};

void init_version(py::module_ &m) {
    // bind global version information
    // complexity necessary to enforce read-only
    py::class_<version>(m, "version")
        .def_property_readonly_static("name", [](py::object /* self */) { return plssvm::version::name; }, "the name of the PLSSVM library")
        .def_property_readonly_static("version", [](py::object /* self */) { return plssvm::version::version; }, "the used version of the PLSSVM library")
        .def_property_readonly_static("major", [](py::object /* self */) { return plssvm::version::major; }, "the used major version of the PLSSVM library")
        .def_property_readonly_static("minor", [](py::object /* self */) { return plssvm::version::minor; }, "the used minor version of the PLSSVM library")
        .def_property_readonly_static("patch", [](py::object /* self */) { return plssvm::version::patch; }, "the used patch version of the PLSSVM library");
}
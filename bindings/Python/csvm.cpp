#include "plssvm/csvm.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_csvm(py::module &m) {
    // TODO: remove type restriction

    py::class_<plssvm::csvm>(m, "csvm")
        .def("get_params", &plssvm::csvm::get_params)
        //        .def("set_params", &plssvm::csvm::set_params) // TODO: implement
        .def("fit", &plssvm::csvm::fit<double, int>)
        .def("predict", &plssvm::csvm::predict<double, int>)  // TODO: variadic template
        .def("score", py::overload_cast<const plssvm::model<double, int> &>(&plssvm::csvm::score<double, int>, py::const_))
        .def("score", py::overload_cast<const plssvm::model<double, int> &, const plssvm::data_set<double, int> &>(&plssvm::csvm::score<double, int>, py::const_));
}
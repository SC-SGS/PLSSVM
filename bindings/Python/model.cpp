#include "plssvm/model.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_model(py::module &m) {
    // TODO: make std::vector<T> opaque? (https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html)
    // bind model class
    using model_type = plssvm::model<double, int>; // TODO: remove type restriction

    py::class_<model_type>(m, "model")
        .def(py::init<const std::string &>())
        .def("save", &model_type::save)
        .def("num_support_vectors", &model_type::num_support_vectors)
        .def("num_features", &model_type::num_features)
        .def("get_params", &model_type::get_params, py::return_value_policy::reference)
        .def("support_vectors", &model_type::support_vectors, py::return_value_policy::reference)
        .def("weights", &model_type::weights, py::return_value_policy::reference)
        .def("rho", &model_type::rho);
}
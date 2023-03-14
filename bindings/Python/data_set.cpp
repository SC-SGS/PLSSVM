#include "plssvm/data_set.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_data_set(py::module &m) {
    // TODO: make std::vector<T> opaque? (https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html)
    // bind data_set class
    using data_set_type = plssvm::data_set<double, int>;  // TODO: remove type restriction
    using size_type = typename data_set_type::size_type;

    // TODO: named args?

    py::class_<data_set_type::scaling::factors>(m, "data_set_scaling_factors")
        .def(py::init<size_type, double, double>())
        .def_readonly("feature", &data_set_type::scaling::factors::feature)
        .def_readonly("lower", &data_set_type::scaling::factors::lower)
        .def_readonly("upper", &data_set_type::scaling::factors::upper);

    py::class_<data_set_type::scaling>(m, "data_set_scaling")
        .def(py::init<double, double>())
        .def(py::init<const std::string &>())
        .def("save", &data_set_type::scaling::save)
        .def_readonly("scaling_interval", &data_set_type::scaling::scaling_interval)
        .def_readonly("scaling_factors", &data_set_type::scaling::scaling_factors);

    py::class_<data_set_type>(m, "data_set")
        .def(py::init<const std::string &>())
        .def(py::init<const std::string &, plssvm::file_format_type>())
        .def(py::init<const std::string &, data_set_type::scaling>())
        .def(py::init<const std::string &, plssvm::file_format_type, data_set_type::scaling>())
        .def(py::init<std::vector<std::vector<double>>>())
        .def(py::init<std::vector<std::vector<double>>, std::vector<int>>())
        .def(py::init<std::vector<std::vector<double>>, data_set_type::scaling>())
        .def(py::init<std::vector<std::vector<double>>, std::vector<int>, data_set_type::scaling>())
        .def("save", &data_set_type::save)
        .def("data", &data_set_type::data, py::return_value_policy::reference)
        .def("has_labels", &data_set_type::has_labels)
        .def("labels", &data_set_type::labels)
        .def("different_labels", &data_set_type::different_labels)
        .def("num_data_points", &data_set_type::num_data_points)
        .def("num_features", &data_set_type::num_features)
        .def("num_different_labels", &data_set_type::num_different_labels)
        .def("is_scaled", &data_set_type::is_scaled)
        .def("scaling_factors", &data_set_type::scaling_factors);
}
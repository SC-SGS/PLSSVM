#include "plssvm/data_set.hpp"

#include "pybind11/pybind11.h"  // py::module, py::class_, py::init, py::return_value_policy
#include "pybind11/stl.h"       // support for STL types

#include <string>  // std::string
#include <vector>  // std::vector

namespace py = pybind11;

void init_data_set(py::module &m) {
    // bind data_set class
    using real_type = double;
    using label_type = std::string;
    using data_set_type = plssvm::data_set<real_type, label_type>;
    using size_type = typename data_set_type::size_type;

    // bind the plssvm::data_set::scaling internal "factors" struct
    py::class_<data_set_type::scaling::factors>(m, "data_set_scaling_factors")
        .def(py::init<size_type, real_type, real_type>(), py::arg("feature"), py::arg("lower"), py::arg("upper"))
        .def_readonly("feature", &data_set_type::scaling::factors::feature)
        .def_readonly("lower", &data_set_type::scaling::factors::lower)
        .def_readonly("upper", &data_set_type::scaling::factors::upper);

    // bind the plssvm::data_set internal "scaling" struct
    py::class_<data_set_type::scaling>(m, "data_set_scaling")
        .def(py::init<real_type, real_type>(), py::arg("lower"), py::arg("upper"))
        .def(py::init<const std::string &>())
        .def("save", &data_set_type::scaling::save)
        .def_readonly("scaling_interval", &data_set_type::scaling::scaling_interval)
        .def_readonly("scaling_factors", &data_set_type::scaling::scaling_factors);

    // bind the data set class
    py::class_<data_set_type>(m, "data_set")
        .def(py::init([](const std::string &file_name, py::kwargs args) {
            // check for valid keys
            constexpr static std::array valid_keys = { "file_format", "scaling" };
            for (const auto &[key, value] : args) {
                if (!plssvm::detail::contains(valid_keys, key.cast<std::string>())) {
                    throw py::value_error(fmt::format("Invalid argument \"{}={}\" provided!", key.cast<std::string>(), value.cast<std::string>()));
                }
            }

            // call the constructor corresponding to the provided named arguments
            if (args.contains("file_format") && args.contains("scaling")) {
                return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>(), std::move(args["scaling"].cast<data_set_type::scaling>()) };
            } else if (args.contains("file_format")) {
                return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>() };
            } else if (args.contains("scaling")) {
                return data_set_type{ file_name, std::move(args["scaling"].cast<data_set_type::scaling>()) };
            } else {
                return data_set_type{ file_name };
            }
        }))
        .def(py::init([](std::vector<std::vector<real_type>> data, py::list labels, std::optional<data_set_type::scaling> scaling) {
            std::vector<std::string> tmp(py::len(labels));
            #pragma omp parallel for
            for (std::size_t i = 0; i < py::len(labels); ++i) {
                tmp[i] = labels[i].cast<py::str>().cast<std::string>();
            }
            if (scaling.has_value()) {
                return data_set_type{ std::move(data), std::move(tmp), scaling.value() };
            } else {
                return data_set_type{ std::move(data), std::move(tmp) };
            }
        }), py::arg("data"), py::arg("labels"), py::pos_only(), py::arg("scaling") = std::nullopt)
        .def(py::init([](std::vector<std::vector<real_type>> data, std::optional<data_set_type::scaling> scaling) {
            if (scaling.has_value()) {
                return data_set_type{ std::move(data), scaling.value() };
            } else {
                return data_set_type{ std::move(data) };
            }
        }), py::arg("data"), py::pos_only(), py::arg("scaling") = std::nullopt)
        .def("save", &data_set_type::save)
        .def("data", &data_set_type::data, py::return_value_policy::reference_internal)
        .def("has_labels", &data_set_type::has_labels)
        .def("labels", &data_set_type::labels, py::return_value_policy::reference_internal)
        .def("different_labels", &data_set_type::different_labels)
        .def("num_data_points", &data_set_type::num_data_points)
        .def("num_features", &data_set_type::num_features)
        .def("num_different_labels", &data_set_type::num_different_labels)
        .def("is_scaled", &data_set_type::is_scaled)
        .def("scaling_factors", &data_set_type::scaling_factors, py::return_value_policy::reference_internal);
}
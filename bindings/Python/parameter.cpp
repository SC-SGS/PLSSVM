#include "plssvm/parameter.hpp"

#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module, py::enum_
#include "pybind11/stl.h"        // support for STL types

#include <sstream>

namespace py = pybind11;

void init_parameter(py::module &m) {
    const plssvm::parameter default_params{};

    //    py::class_<plssvm::default_init<int>>(m, "default_init")
    //        .def(py::init<>())
    //        .def(py::init<int>())
    //        .def("__repr__", [](const plssvm::default_init<int> &d) { return std::to_string(d.value); });
    //
    //    py::class_<plssvm::default_value<int>>(m, "default_value")
    //        .def(py::init<>())
    //        .def(py::init<plssvm::default_init<int>>())
    //        .def("assign", [](int value, plssvm::default_value<int> &self) { self = value; })
    //        .def("__repr__", [](const plssvm::default_value<int> &d) { return std::to_string(d.value()); });

    py::class_<plssvm::parameter>(m, "parameter")
        .def(py::init<>())
        .def(py::init<plssvm::kernel_function_type, int, double, double, double>(),
             py::arg("kernel_type") = default_params.kernel_type.value(),
             py::arg("degree") = default_params.degree.value(),
             py::arg("gamma") = default_params.gamma.value(),
             py::arg("coef0") = default_params.coef0.value(),
             py::arg("cost") = default_params.cost.value())
        .def("equivalent", &plssvm::parameter::equivalent)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", [](const plssvm::parameter &param) {
            std::ostringstream os;
            os << param;
            return os.str();
        });
    // TODO: add direct member access

//        .def_property(
//            "degree", [](plssvm::parameter &param, int degree) { param.degree = degree; }, [](const plssvm::parameter &param) { return param.degree; });
    //                .def_readwrite("degree", &plssvm::parameter::degree);

    // bind free functions
    m.def("equivalent", &plssvm::detail::equivalent<double>);
}

//.def(py::init([](const std::string &file_name, py::kwargs args) {
//    // check for valid keys
//    constexpr static std::array valid_keys = { "file_format", "scaling" };
//    for (const auto &[key, value] : args) {
//        if (!plssvm::detail::contains(valid_keys, key.cast<std::string>())) {
//            throw py::value_error(fmt::format("Invalid argument \"{}={}\" provided!", key.cast<std::string>(), value.cast<std::string>()));
//        }
//    }
//
//    // call the constructor corresponding to the provided named arguments
//    if (args.contains("file_format") && args.contains("scaling")) {
//        return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>(), std::move(args["scaling"].cast<data_set_type::scaling>()) };
//    } else if (args.contains("file_format")) {
//        return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>() };
//    } else if (args.contains("scaling")) {
//        return data_set_type{ file_name, std::move(args["scaling"].cast<data_set_type::scaling>()) };
//    } else {
//        return data_set_type{ file_name };
//    }
//}))
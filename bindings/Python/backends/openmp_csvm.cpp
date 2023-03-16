#include "plssvm/backends/OpenMP/csvm.hpp"

#include "plssvm/detail/utility.hpp"

#include "fmt/core.h"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_openmp_csvm(py::module &m) {
    // TODO: remove type restriction
    // TODO: own module?
    // TODO: only if OpenMP backend is available

    py::class_<plssvm::openmp::csvm, plssvm::csvm>(m, "openmp_csvm")
        .def(py::init<>())
        .def(py::init<plssvm::target_platform>())
        .def(py::init<plssvm::parameter>())
        .def(py::init<plssvm::target_platform, plssvm::parameter>())
        .def(py::init([](py::kwargs args) {
            // check for valid keys
            constexpr static std::array valid_keys = { "target_platform", "kernel_type", "degree", "gamma", "coef0", "cost" };
            for (const auto &[key, value] : args) {
                if (!plssvm::detail::contains(valid_keys, key.cast<std::string>())) {
                    throw py::value_error(fmt::format("Invalid argument \"{}={}\" provided!", key.cast<std::string>(), value.cast<std::string>()));
                }
            }

            // if one of the value named parameter is provided, set the respective value
            plssvm::parameter params{};
            if (args.contains("kernel_type")) {
                params.kernel_type = args["kernel_type"].cast<plssvm::kernel_function_type>();
            }
            if (args.contains("degree")) {
                params.degree = args["degree"].cast<int>();
            }
            if (args.contains("gamma")) {
                params.gamma = args["gamma"].cast<double>();
            }
            if (args.contains("coef0")) {
                params.coef0 = args["coef0"].cast<double>();
            }
            if (args.contains("cost")) {
                params.cost = args["cost"].cast<double>();
            }

            // TODO: necessary for the Python API?
            if (args.contains("target_platform")) {
                return std::make_unique<plssvm::openmp::csvm>(args["target_platform"].cast<plssvm::target_platform>(), params);
            } else {
                return std::make_unique<plssvm::openmp::csvm>(params);
            }
        }));
}
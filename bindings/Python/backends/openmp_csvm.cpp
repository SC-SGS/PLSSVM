#include "plssvm/backends/OpenMP/csvm.hpp"
#include "plssvm/backends/OpenMP/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_openmp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the OpenMP CSVM bindings
    py::module_ openmp_module = m.def_submodule("openmp", "a module containing all OpenMP backend specific functionality");

    // bind the CSVM using the OpenMP backend
    py::class_<plssvm::openmp::csvm, plssvm::csvm>(openmp_module, "Csvm")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameters")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the default parameters")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create a new SVM with the provided target platform and parameters")
        .def(py::init([](py::kwargs args) {
            // check for valid keys
            check_kwargs_for_correctness(args, { "target_platform", "kernel_type", "degree", "gamma", "coef0", "cost" });

            // if one of the value named parameter is provided, set the respective value
            const plssvm::parameter params = convert_kwargs_to_parameter(args);

            if (args.contains("target_platform")) {
                return std::make_unique<plssvm::openmp::csvm>(args["target_platform"].cast<plssvm::target_platform>(), params);
            } else {
                return std::make_unique<plssvm::openmp::csvm>(params);
            }
        }), "create an SVM using keyword arguments");

    // register OpenMP backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::openmp::backend_exception, openmp_module, BackendError, base_exception)
}
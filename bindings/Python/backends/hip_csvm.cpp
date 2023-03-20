#include "plssvm/backends/HIP/csvm.hpp"
#include "plssvm/backends/HIP/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_hip_csvm(py::module_ &m) {
    // use its own submodule for the HIP CSVM bindings
    py::module_ hip_module = m.def_submodule("hip", "a module containing all HIP backend specific functionality");

    // bind the CSVM using the HIP backend
    py::class_<plssvm::hip::csvm, plssvm::csvm>(hip_module, "Csvm")
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
                return std::make_unique<plssvm::hip::csvm>(args["target_platform"].cast<plssvm::target_platform>(), params);
            } else {
                return std::make_unique<plssvm::hip::csvm>(params);
            }
        }), "create an SVM using keyword arguments");

    // register HIP backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::hip::backend_exception, hip_module, BackendError)
}
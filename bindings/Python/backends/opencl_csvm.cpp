#include "plssvm/backends/OpenCL/csvm.hpp"
#include "plssvm/backends/OpenCL/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_opencl_csvm(py::module_ &m) {
    // use its own submodule for the OpenCL CSVM bindings
    py::module_ opencl_module = m.def_submodule("opencl", "a module containing all OpenCL backend specific functionality");

    // bind the CSVM using the OpenCL backend
    py::class_<plssvm::opencl::csvm, plssvm::csvm>(opencl_module, "Csvm")
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
                return std::make_unique<plssvm::opencl::csvm>(args["target_platform"].cast<plssvm::target_platform>(), params);
            } else {
                return std::make_unique<plssvm::opencl::csvm>(params);
            }
        }), "create an SVM using keyword arguments");

    // register OpenCL backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::opencl::backend_exception, opencl_module, BackendError)
}
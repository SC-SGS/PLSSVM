#include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"
#include "plssvm/backends/SYCL/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_hipsycl_csvm(py::module &m) {
    // use its own submodule for the hipSYCL CSVM bindings
    py::module hipsycl_module = m.def_submodule("hipsycl");

    // bind the CSVM using the hipSYCL backend
    py::class_<plssvm::hipsycl::csvm, plssvm::csvm>(hipsycl_module, "csvm")
        .def(py::init<>())
        .def(py::init<plssvm::target_platform>())
        .def(py::init<plssvm::parameter>())
        .def(py::init<plssvm::target_platform, plssvm::parameter>())
        .def(py::init([](py::kwargs args) {
            // check for valid keys
            check_kwargs_for_correctness(args, { "target_platform", "kernel_type", "degree", "gamma", "coef0", "cost" });

            // if one of the value named parameter is provided, set the respective value
            const plssvm::parameter params = convert_kwargs_to_parameter(args);

            if (args.contains("target_platform")) {
                return std::make_unique<plssvm::hipsycl::csvm>(args["target_platform"].cast<plssvm::target_platform>(), params);
            } else {
                return std::make_unique<plssvm::hipsycl::csvm>(params);
            }
        }));

    // register hipSYCL backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::hipsycl::backend_exception, hipsycl_module, backend_error)
}
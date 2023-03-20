#include "plssvm/backends/SYCL/exceptions.hpp"
#include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_hipsycl_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the hipSYCL CSVM bindings
    py::module_ hipsycl_module = m.def_submodule("hipsycl", "a module containing all hipSYCL SYCL backend specific functionality");

    // bind the CSVM using the hipSYCL backend
    py::class_<plssvm::hipsycl::csvm, plssvm::csvm>(hipsycl_module, "Csvm")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameters")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the default parameters")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create a new SVM with the provided target platform and parameters")
        .def(py::init([](py::kwargs args) {
            // check for valid keys
            check_kwargs_for_correctness(args, { "target_platform", "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_kernel_invocation_type" });

            // if one of the value named parameter is provided, set the respective value
            const plssvm::parameter params = convert_kwargs_to_parameter(args);

            // set target platform
            const plssvm::target_platform target = args.contains("target_platform") ? args["target_platform"].cast<plssvm::target_platform>() : plssvm::target_platform::automatic;

            if (args.contains("sycl_kernel_invocation_type")) {
                return std::make_unique<plssvm::hipsycl::csvm>(target, params, plssvm::sycl_kernel_invocation_type = args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>());
            } else {
                return std::make_unique<plssvm::hipsycl::csvm>(target, params);
            }
        }), "create an SVM using keyword arguments");

    // register hipSYCL backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::hipsycl::backend_exception, hipsycl_module, BackendError, base_exception)
}
#include "plssvm/backends/SYCL/exceptions.hpp"
#include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

py::module_ init_hipsycl_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the hipSYCL CSVM bindings
    py::module_ hipsycl_module = m.def_submodule("hipsycl", "a module containing all hipSYCL SYCL backend specific functionality");

    // bind the CSVM using the hipSYCL backend
    py::class_<plssvm::hipsycl::csvm, plssvm::csvm>(hipsycl_module, "CSVM")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameter object")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform and provided parameter object")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the provided target platform and default parameter object")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create an SVM with the provided target platform and parameter object")
        .def(py::init([](py::kwargs args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_kernel_invocation_type" });
                 // if one of the value named parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // set SYCL kernel invocation type
                 const plssvm::sycl::kernel_invocation_type invoc = args.contains("sycl_kernel_invocation_type") ? args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>() : plssvm::sycl::kernel_invocation_type::automatic;
                 // create CSVM with the default target platform
                 return std::make_unique<plssvm::hipsycl::csvm>(params, plssvm::sycl_kernel_invocation_type = invoc);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, py::kwargs args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_kernel_invocation_type" });
                 // if one of the value named parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // set SYCL kernel invocation type
                 const plssvm::sycl::kernel_invocation_type invoc = args.contains("sycl_kernel_invocation_type") ? args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>() : plssvm::sycl::kernel_invocation_type::automatic;
                 // create CSVM with the default target platform
                 return std::make_unique<plssvm::hipsycl::csvm>(target, params, plssvm::sycl_kernel_invocation_type = invoc);
             }),
             "create an SVM with the provided target platform and keyword arguments");

    // register hipSYCL backend specific exceptions
    register_py_exception<plssvm::hipsycl::backend_exception>(hipsycl_module, "BackendError", base_exception);

    return hipsycl_module;
}
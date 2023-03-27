#include "plssvm/backends/CUDA/csvm.hpp"
#include "plssvm/backends/CUDA/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_cuda_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the CUDA CSVM bindings
    py::module_ cuda_module = m.def_submodule("cuda", "a module containing all CUDA backend specific functionality");

    // bind the CSVM using the CUDA backend
    py::class_<plssvm::cuda::csvm, plssvm::csvm>(cuda_module, "CSVM")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameter object")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform and provided parameter object")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the provided target platform and default parameter object")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create an SVM with the provided target platform and parameter object")
        .def(py::init([](py::kwargs args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value named parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the default target platform
                 return std::make_unique<plssvm::cuda::csvm>(params);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, py::kwargs args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value named parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the provided target platform
                 return std::make_unique<plssvm::cuda::csvm>(target, params);
             }),
             "create an SVM with the provided target platform and keyword arguments");

    // register CUDA backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::cuda::backend_exception, cuda_module, BackendError, base_exception)
}
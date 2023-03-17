#include "plssvm/backends/CUDA/csvm.hpp"
#include "plssvm/backends/CUDA/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_cuda_csvm(py::module &m) {
    // use its own submodule for the CUDA CSVM bindings
    py::module cuda_module = m.def_submodule("cuda");

    // bind the CSVM using the CUDA backend
    py::class_<plssvm::cuda::csvm, plssvm::csvm>(cuda_module, "csvm")
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
                return std::make_unique<plssvm::cuda::csvm>(args["target_platform"].cast<plssvm::target_platform>(), params);
            } else {
                return std::make_unique<plssvm::cuda::csvm>(params);
            }
        }));

    // register CUDA backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::cuda::backend_exception, cuda_module, backend_error)
}
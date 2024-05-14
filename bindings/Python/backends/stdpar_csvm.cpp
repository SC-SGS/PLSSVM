/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/csvm.hpp"                  // plssvm::stdpar::csvm
#include "plssvm/backends/stdpar/exceptions.hpp"            // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/implementation_types.hpp"  // plssvm::stdpar::implementation_type
#include "plssvm/csvm.hpp"                                  // plssvm::csvm
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/pytypes.h"   // py::kwargs

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_stdpar_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the stdpar CSVM bindings
    py::module_ stdpar_module = m.def_submodule("stdpar", "a module containing all stdpar backend specific functionality");

    // bind the enum class
    py::enum_<plssvm::stdpar::implementation_type>(stdpar_module, "ImplementationType")
        .value("NVHPC", plssvm::stdpar::implementation_type::nvhpc, "use NVIDIA's HPC SDK (NVHPC) compiler nvc++")
        .value("ROC_STDPAR", plssvm::stdpar::implementation_type::roc_stdpar, "use AMD's roc-stdpar compiler (patched LLVM)")
        .value("INTEL_LLVM", plssvm::stdpar::implementation_type::intel_llvm, "use Intel's LLVM compiler icpx")
        .value("ADAPTIVECPP", plssvm::stdpar::implementation_type::adaptivecpp, "use AdaptiveCpp (formerly known as hipSYCL)")
        .value("GNU_TBB", plssvm::stdpar::implementation_type::gnu_tbb, "use GNU GCC + Intel's TBB library");

    // bind the CSVM using the stdpar backend
    py::class_<plssvm::stdpar::csvm, plssvm::csvm>(stdpar_module, "CSVM")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameter object")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform and provided parameter object")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the provided target platform and default parameter object")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create an SVM with the provided target platform and parameter object")
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the default target platform
                 return std::make_unique<plssvm::stdpar::csvm>(params);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the provided target platform
                 return std::make_unique<plssvm::stdpar::csvm>(target, params);
             }),
             "create an SVM with the provided target platform and keyword arguments")
        .def("get_implementation_type", &plssvm::stdpar::csvm::get_implementation_type, "get the stdpar implementation type used in this stdpar SVM")
        .def("num_available_devices", &plssvm::stdpar::csvm::num_available_devices, "the number of available devices");

    // register stdpar backend specific exceptions
    register_py_exception<plssvm::stdpar::backend_exception>(stdpar_module, "BackendError", base_exception);
}

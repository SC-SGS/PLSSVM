/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/csvm.hpp"             // plssvm::kokkos::csvm
#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space
#include "plssvm/csvm.hpp"                             // plssvm::csvm
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/parameter.hpp"                        // plssvm::parameter, plssvm::kokkos_execution_space
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_kokkos_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the Kokkos CSVM bindings
    py::module_ kokkos_module = m.def_submodule("kokkos", "a module containing all Kokkos backend specific functionality");

    // bind the CSVM using the Kokkos backend
    py::class_<plssvm::kokkos::csvm, plssvm::csvm>(kokkos_module, "CSVM")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameter object")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform and provided parameter object")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the provided target platform and default parameter object")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create an SVM with the provided target platform and parameter object")
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "kokkos_execution_space" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // set Kokkos execution space
                 const plssvm::kokkos::execution_space space = args.contains("kokkos_execution_space") ? args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>() : plssvm::kokkos::execution_space::automatic;
                 // create CSVM with the default target platform
                 return std::make_unique<plssvm::kokkos::csvm>(params, plssvm::kokkos_execution_space = space);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "kokkos_execution_space" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // set Kokkos execution space
                 const plssvm::kokkos::execution_space space = args.contains("kokkos_execution_space") ? args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>() : plssvm::kokkos::execution_space::automatic;
                 // create CSVM with the provided target platform
                 return std::make_unique<plssvm::kokkos::csvm>(target, params, plssvm::kokkos_execution_space = space);
             }),
             "create an SVM with the provided target platform and keyword arguments")
        .def("get_execution_space", &plssvm::kokkos::csvm::get_execution_space, "get the Kokkos execution space used in this Kokkos SVM");

    // register Kokkos backend specific exceptions
    register_py_exception<plssvm::kokkos::backend_exception>(kokkos_module, "BackendError", base_exception);

    // bind the execution space enum classes
    py::enum_<plssvm::kokkos::execution_space>(kokkos_module, "ExecutionSpace")
        .value("AUTOMATIC", plssvm::kokkos::execution_space::cuda, "automatically determine the used Kokkos execution space (note: this does not necessarily correspond to Kokkos::DefaultExecutionSpace)")
        .value("CUDA", plssvm::kokkos::execution_space::cuda, "execution space representing execution on a CUDA device")
        .value("HIP", plssvm::kokkos::execution_space::hip, "execution space representing execution on a device supported by HIP")
        .value("SYCL", plssvm::kokkos::execution_space::sycl, "execution space representing execution on a device supported by SYCL")
        .value("HPX", plssvm::kokkos::execution_space::hpx, "execution space representing execution with the HPX runtime system")
        .value("OPENMP", plssvm::kokkos::execution_space::openmp, "execution space representing execution with the OpenMP runtime system")
        .value("OPENMPTARGET", plssvm::kokkos::execution_space::openmp_target, "execution space representing execution using the target offloading feature of the OpenMP runtime system")
        .value("OPENACC", plssvm::kokkos::execution_space::openacc, "execution space representing execution with the OpenACC runtime system")
        .value("THREADS", plssvm::kokkos::execution_space::threads, "execution space representing parallel execution with std::threads")
        .value("SERIAL", plssvm::kokkos::execution_space::serial, "execution space representing serial execution on the CPU; should always be available");

    kokkos_module.def("list_available_execution_spaces", &plssvm::kokkos::list_available_execution_spaces, "list all available Kokkos execution spaces");
}

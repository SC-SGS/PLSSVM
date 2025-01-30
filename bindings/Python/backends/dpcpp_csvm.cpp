/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"                          // plssvm::dpcpp::backend_csvm_type_t
#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"               // plssvm::dpcpp::csvm
#include "plssvm/backends/SYCL/exceptions.hpp"               // plssvm::dpcpp::backend_exception
#include "plssvm/backends/SYCL/kernel_invocation_types.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/exceptions/exceptions.hpp"                  // plssvm::exception
#include "plssvm/parameter.hpp"                              // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                               // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                               // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                               // plssvm::csvr
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::exception
#include "pybind11/pytypes.h"   // py::kwargs

#include <memory>  // std::make_unique
#include <string>  // std::string

namespace py = pybind11;

namespace {

template <typename csvm_type>
void bind_dpcpp_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::dpcpp::backend_csvm_type_t<csvm_type>;

    // assemble docstrings
    const std::string param_docstring{ fmt::format("create a DPC++ SYCL {} with the provided parameters and optional SYCL specific keyword arguments", csvm_name) };
    const std::string target_param_docstring{ fmt::format("create a DPC++ SYCL {} with the provided target platform, parameters, and optional SYCL specific keyword arguments", csvm_name) };
    const std::string kwargs_docstring{ fmt::format("create a DPC++ SYCL {} with the provided keyword arguments (including optional SYCL specific keyword arguments)", csvm_name) };
    const std::string target_kwargs_docstring{ fmt::format("create a DPC++ SYCL {} with the provided target platform and keyword arguments (including optional SYCL specific keyword arguments)", csvm_name) };

    py::class_<backend_csvm_type, plssvm::dpcpp::csvm, csvm_type>(m, csvm_name.c_str())
        .def(py::init([](const plssvm::parameter params, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "sycl_kernel_invocation_type" });
                 // set SYCL kernel invocation type
                 const plssvm::sycl::kernel_invocation_type invocation = args.contains("sycl_kernel_invocation_type") ? args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>() : plssvm::sycl::kernel_invocation_type::automatic;
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(params, plssvm::sycl_kernel_invocation_type = invocation);
             }),
             param_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const plssvm::parameter params, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "sycl_kernel_invocation_type" });
                 // set SYCL kernel invocation type
                 const plssvm::sycl::kernel_invocation_type invocation = args.contains("sycl_kernel_invocation_type") ? args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>() : plssvm::sycl::kernel_invocation_type::automatic;
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(target, params, plssvm::sycl_kernel_invocation_type = invocation);
             }),
             target_param_docstring.c_str())
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_kernel_invocation_type" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // set SYCL kernel invocation type
                 const plssvm::sycl::kernel_invocation_type invocation = args.contains("sycl_kernel_invocation_type") ? args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>() : plssvm::sycl::kernel_invocation_type::automatic;
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(params, plssvm::sycl_kernel_invocation_type = invocation);
             }),
             kwargs_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_kernel_invocation_type" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // set SYCL kernel invocation type
                 const plssvm::sycl::kernel_invocation_type invocation = args.contains("sycl_kernel_invocation_type") ? args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>() : plssvm::sycl::kernel_invocation_type::automatic;
                 // create C-SVM with the provided target platform
                 return std::make_unique<backend_csvm_type>(target, params, plssvm::sycl_kernel_invocation_type = invocation);
             }),
             target_kwargs_docstring.c_str())
        .def("get_kernel_invocation_type", &plssvm::dpcpp::csvm::get_kernel_invocation_type, "get the kernel invocation type used in this SYCL C-SVM");
}

}  // namespace

py::module_ init_dpcpp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the DPC++ C-SVM bindings
    py::module_ dpcpp_module = m.def_submodule("dpcpp", "a module containing all DPC++ backend specific functionality");
    const py::module_ dpcpp_pure_virtual_module = dpcpp_module.def_submodule("__pure_virtual", "a module containing all pure-virtual DPC++ backend specific functionality");

    // bind the pure-virtual base DPC++ C-SVM
    [[maybe_unused]] const py::class_<plssvm::dpcpp::csvm, plssvm::csvm> virtual_base_dpcpp_csvm(dpcpp_pure_virtual_module, "__pure_virtual_dpcpp_base_CSVM");

    // bind the specific DPC++ C-SVC and C-SVR classes
    bind_dpcpp_csvms<plssvm::csvc>(dpcpp_module, "CSVC");
    bind_dpcpp_csvms<plssvm::csvr>(dpcpp_module, "CSVR");

    // register DPC++ backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::dpcpp::backend_exception>(dpcpp_module, "BackendError", base_exception);

    return dpcpp_module;
}

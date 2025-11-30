/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"                    // plssvm::kokkos::backend_csvm_type_t
#include "plssvm/backends/Kokkos/csvm.hpp"             // plssvm::kokkos::csvm
#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_spaces.hpp"  // plssvm::kokkos::execution_space
#include "plssvm/constants.hpp"                        // plssvm::real_type
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/gamma.hpp"                            // plssvm::gamma
#include "plssvm/kernel_function_types.hpp"            // plssvm::kernel_function_type
#include "plssvm/mpi/communicator.hpp"                 // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                         // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                         // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                         // plssvm::csvr
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "bindings/Python/bindings_fwd.hpp"                 // forward declare all helper functions to create the Python bindings
#include "bindings/Python/type_caster/mpi_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"                      // plssvm::bindings::python::util::register_py_exception

#include "fmt/format.h"            // fmt::format
#include "pybind11/native_enum.h"  // py::native_enum
#include "pybind11/pybind11.h"     // py::module_, py::class_, py::init, py::arg, py::exception, py::module_local
#include "pybind11/stl.h"          // support for STL types: std::variant

#include <memory>   // std::make_unique
#include <string>   // std::string
#include <utility>  // std::move

namespace py = pybind11;

namespace {

template <typename csvm_type>
void bind_kokkos_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::kokkos::backend_csvm_type_t<csvm_type>;

    // the default parameters used
    const plssvm::parameter default_params{};

    // assemble docstrings
    const std::string class_docstring{ fmt::format("A {} using the Kokkos backend.", csvm_name) };
    const std::string params_constructor_docstring{ fmt::format("create a Kokkos {} with the provided SVM parameter encapsulated in a plssvm.Parameter and optional Kokkos specific keyword arguments", csvm_name) };
    const std::string keyword_args_constructor_docstring{ fmt::format("create a Kokkos {} with the provided SVM parameter as separate keyword arguments including optional Kokkos specific keyword arguments", csvm_name) };

    py::class_<backend_csvm_type, plssvm::kokkos::csvm, csvm_type>(m, csvm_name.c_str(), class_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const plssvm::parameter params, const plssvm::kokkos::execution_space space, plssvm::mpi::communicator comm) {
                 return std::make_unique<backend_csvm_type>(std::move(comm), target, params, plssvm::kokkos_execution_space = space);
             }),
             params_constructor_docstring.c_str(),
             py::arg("target") = plssvm::target_platform::automatic,
             py::kw_only(),
             py::arg("params") = default_params,
             py::arg("kokkos_execution_space") = plssvm::kokkos::execution_space::automatic,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](const plssvm::target_platform target, const plssvm::kernel_function_type kernel_type, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0, const plssvm::real_type cost, const plssvm::kokkos::execution_space space, plssvm::mpi::communicator comm) {
                 const plssvm::parameter params{ kernel_type, degree, gamma, coef0, cost };
                 return std::make_unique<backend_csvm_type>(std::move(comm), target, params, plssvm::kokkos_execution_space = space);
             }),
             keyword_args_constructor_docstring.c_str(),
             py::arg("target") = plssvm::target_platform::automatic,
             py::kw_only(),
             py::arg("kernel_type") = default_params.kernel_type,
             py::arg("degree") = default_params.degree,
             py::arg("gamma") = default_params.gamma,
             py::arg("coef0") = default_params.coef0,
             py::arg("cost") = default_params.cost,
             py::arg("kokkos_execution_space") = plssvm::kokkos::execution_space::automatic,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def("get_execution_space", &plssvm::kokkos::csvm::get_execution_space, "get the Kokkos execution space used in this Kokkos C-SVM")
        .def("__repr__", [csvm_name](const backend_csvm_type &self) {
            return fmt::format("<plssvm.kokkos.{} with {{ #devices: {}, execution_space: {} }}>", csvm_name, self.num_available_devices(), self.get_execution_space());
        });
}

}  // namespace

void init_kokkos_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the Kokkos C-SVM bindings
    py::module_ kokkos_module = m.def_submodule("kokkos", "a module containing all Kokkos backend specific functionality");

    // bind the enum class
    py::native_enum<plssvm::kokkos::execution_space> py_enum(kokkos_module, "ExecutionSpace", "enum.Enum", "Enum class for all supported Kokkos execution spaces in PLSSVM.");
    py_enum
        .value("AUTOMATIC", plssvm::kokkos::execution_space::automatic, "automatically determine the used Kokkos execution space; note: this does not necessarily correspond to Kokkos::DefaultExecutionSpace!")
        .value("CUDA", plssvm::kokkos::execution_space::cuda, "execution space representing execution on a CUDA device")
        .value("HIP", plssvm::kokkos::execution_space::hip, "execution space representing execution on a device supported by HIP")
        .value("SYCL", plssvm::kokkos::execution_space::sycl, "execution space representing execution on a device supported by SYCL")
        .value("HPX", plssvm::kokkos::execution_space::hpx, "execution space representing execution with the HPX runtime system")
        .value("OPENMP", plssvm::kokkos::execution_space::openmp, "execution space representing execution with the OpenMP runtime system")
        .value("OPENMPTARGET", plssvm::kokkos::execution_space::openmp_target, "execution space representing execution using the target offloading feature of the OpenMP runtime system")
        .value("OPENACC", plssvm::kokkos::execution_space::openacc, "execution space representing execution with the OpenACC runtime system")
        .value("THREADS", plssvm::kokkos::execution_space::threads, "execution space representing parallel execution with std::threads")
        .value("SERIAL", plssvm::kokkos::execution_space::serial, "execution space representing serial execution on the CPU. Should always be available")
        .finalize();

    // bind the pure-virtual base Kokkos C-SVM
    [[maybe_unused]] const py::class_<plssvm::kokkos::csvm, plssvm::csvm> virtual_base_kokkos_csvm(m, "__pure_virtual_kokkos_CSVM", py::module_local());

    // bind the specific Kokkos C-SVC and C-SVR classes
    bind_kokkos_csvms<plssvm::csvc>(kokkos_module, "CSVC");
    bind_kokkos_csvms<plssvm::csvr>(kokkos_module, "CSVR");

    // register Kokkos backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::kokkos::backend_exception>(kokkos_module, "BackendError", base_exception);
}

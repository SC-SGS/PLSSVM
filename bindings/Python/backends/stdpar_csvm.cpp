/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"                         // plssvm::stdpar::backend_csvm_type_t
#include "plssvm/backends/stdpar/csvm.hpp"                  // plssvm::stdpar::csvm
#include "plssvm/backends/stdpar/exceptions.hpp"            // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/implementation_types.hpp"  // plssvm::stdpar::implementation_type
#include "plssvm/constants.hpp"                             // plssvm::real_type
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/gamma.hpp"                                 // plssvm::gamma
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
#include "plssvm/mpi/communicator.hpp"                      // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                              // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                              // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                              // plssvm::csvr
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "bindings/Python/type_caster/mpi_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"                      // plssvm::bindings::python::util::register_py_exception,

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
void bind_stdpar_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::stdpar::backend_csvm_type_t<csvm_type>;

    // the default parameters used
    const plssvm::parameter default_params{};

    // assemble docstrings
    const std::string class_docstring{ fmt::format("A {} using the stdpar backend.", csvm_name) };
    const std::string params_constructor_docstring{ fmt::format("create an stdpar {} with the provided SVM parameter encapsulated in a plssvm.Parameter", csvm_name) };
    const std::string keyword_args_constructor_docstring{ fmt::format("create an stdpar {} with the provided SVM parameter as separate keyword arguments", csvm_name) };

    py::class_<backend_csvm_type, plssvm::stdpar::csvm, csvm_type>(m, csvm_name.c_str(), class_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const plssvm::parameter params, plssvm::mpi::communicator comm) {
                 return std::make_unique<backend_csvm_type>(std::move(comm), target, params);
             }),
             params_constructor_docstring.c_str(),
             py::arg("target") = plssvm::target_platform::automatic,
             py::kw_only(),
             py::arg("params") = default_params,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](const plssvm::target_platform target, const plssvm::kernel_function_type kernel_type, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0, const plssvm::real_type cost, plssvm::mpi::communicator comm) {
                 const plssvm::parameter params{ kernel_type, degree, gamma, coef0, cost };
                 return std::make_unique<backend_csvm_type>(std::move(comm), target, params);
             }),
             keyword_args_constructor_docstring.c_str(),
             py::arg("target") = plssvm::target_platform::automatic,
             py::kw_only(),
             py::arg("kernel_type") = default_params.kernel_type,
             py::arg("degree") = default_params.degree,
             py::arg("gamma") = default_params.gamma,
             py::arg("coef0") = default_params.coef0,
             py::arg("cost") = default_params.cost,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def("get_implementation_type", &plssvm::stdpar::csvm::get_implementation_type, "get the stdpar implementation used in this stdpar C-SVM")
        .def("__repr__", [csvm_name](const backend_csvm_type &self) {
            return fmt::format("<plssvm.stdpar.{} with {{ #devices: {}, implementation_type: {} }}>", csvm_name, self.num_available_devices(), self.get_implementation_type());
        });
}

}  // namespace

void init_stdpar_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the stdpar C-SVM bindings
    py::module_ stdpar_module = m.def_submodule("stdpar", "a module containing all stdpar backend specific functionality");

    // bind the enum class
    py::native_enum<plssvm::stdpar::implementation_type> py_enum(stdpar_module, "ImplementationType", "enum.Enum", "Enum class for all supported stdpar implementations in PLSSVM.");
    py_enum
        .value("NVHPC", plssvm::stdpar::implementation_type::nvhpc, "use NVIDIA's HPC SDK (NVHPC) compiler nvc++")
        .value("ROC_STDPAR", plssvm::stdpar::implementation_type::roc_stdpar, "use AMD's roc-stdpar compiler (patched LLVM)")
        .value("INTEL_LLVM", plssvm::stdpar::implementation_type::intel_llvm, "use Intel's LLVM compiler icpx")
        .value("ADAPTIVECPP", plssvm::stdpar::implementation_type::adaptivecpp, "use AdaptiveCpp (formerly known as hipSYCL)")
        .value("GNU_TBB", plssvm::stdpar::implementation_type::gnu_tbb, "use GNU GCC + Intel's TBB library")
        .finalize();

    stdpar_module.def("list_available_stdpar_implementations", &plssvm::stdpar::list_available_stdpar_implementations, "list all available stdpar implementations");

    // bind the pure-virtual base stdpar C-SVM
    [[maybe_unused]] const py::class_<plssvm::stdpar::csvm, plssvm::csvm> virtual_base_stdpar_csvm(m, "__pure_virtual_stdpar_CSVM", py::module_local());

    // bind the specific stdpar C-SVC and C-SVR classes
    bind_stdpar_csvms<plssvm::csvc>(stdpar_module, "CSVC");
    bind_stdpar_csvms<plssvm::csvr>(stdpar_module, "CSVR");

    // register stdpar backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::stdpar::backend_exception>(stdpar_module, "BackendError", base_exception);
}

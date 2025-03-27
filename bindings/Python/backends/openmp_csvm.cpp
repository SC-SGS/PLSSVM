/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"               // plssvm::openmp::backend_csvm_type_t
#include "plssvm/backends/OpenMP/csvm.hpp"        // plssvm::openmp::csvm
#include "plssvm/backends/OpenMP/exceptions.hpp"  // plssvm::openmp::backend_exception
#include "plssvm/constants.hpp"                   // plssvm::real_type
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/gamma.hpp"                       // plssvm::gamma
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/mpi/communicator.hpp"            // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                    // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                    // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                    // plssvm::csvr
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "bindings/Python/type_caster/mpi_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"                      // plssvm::bindings::python::util::register_py_exception

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::arg, py::exception, py::module_local
#include "pybind11/stl.h"       // support for STL types: std::variant

#include <memory>   // std::make_unique
#include <string>   // std::string
#include <utility>  // std::move

namespace py = pybind11;

namespace {

template <typename csvm_type>
void bind_openmp_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::openmp::backend_csvm_type_t<csvm_type>;

    // the default parameters used
    const plssvm::parameter default_params{};

    // assemble docstrings
    const std::string class_docstring{ fmt::format("A {} using the OpenMP backend.", csvm_name) };
    const std::string params_constructor_docstring{ fmt::format("create an OpenMP {} with the provided SVM parameter encapsulated in a plssvm.Parameter", csvm_name) };
    const std::string keyword_args_constructor_docstring{ fmt::format("create an OpenMP {} with the provided SVM parameter as separate keyword arguments", csvm_name) };

    py::class_<backend_csvm_type, plssvm::openmp::csvm, csvm_type>(m, csvm_name.c_str(), class_docstring.c_str())
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
        .def("__repr__", [csvm_name](const backend_csvm_type &self) {
            return fmt::format("<plssvm.openmp.{} with {{ #devices: {} }}>", csvm_name, self.num_available_devices());
        });
}

}  // namespace

void init_openmp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the OpenMP C-SVM bindings
    py::module_ openmp_module = m.def_submodule("openmp", "a module containing all OpenMP backend specific functionality");

    // bind the pure-virtual base OpenMP C-SVM
    [[maybe_unused]] const py::class_<plssvm::openmp::csvm, plssvm::csvm> virtual_base_openmp_csvm(m, "__pure_virtual_openmp_CSVM", py::module_local());

    // bind the specific OpenMP C-SVC and C-SVR classes
    bind_openmp_csvms<plssvm::csvc>(openmp_module, "CSVC");
    bind_openmp_csvms<plssvm::csvr>(openmp_module, "CSVR");

    // register OpenMP backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::openmp::backend_exception>(openmp_module, "BackendError", base_exception);
}

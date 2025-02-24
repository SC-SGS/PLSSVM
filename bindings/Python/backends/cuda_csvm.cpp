/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"             // plssvm::cuda::backend_csvm_type_t
#include "plssvm/backends/CUDA/csvm.hpp"        // plssvm::cuda::csvm
#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::exception
#include "plssvm/mpi/communicator.hpp"          // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                  // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                  // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                  // plssvm::csvr
#include "plssvm/target_platforms.hpp"          // plssvm::target_platform

#include "bindings/Python/mpi/mpi_typecaster.hpp"  // a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"             // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::exception
#include "pybind11/pytypes.h"   // py::kwargs

#include <memory>   // std::make_unique
#include <string>   // std::string
#include <utility>  // std::move

namespace py = pybind11;

namespace {

template <typename csvm_type>
void bind_cuda_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::cuda::backend_csvm_type_t<csvm_type>;

    // assemble docstrings
    const std::string class_docstring{ fmt::format("A {} using the CUDA backend.", csvm_name) };
    const std::string param_docstring{ fmt::format("create a CUDA {} with the provided parameters", csvm_name) };
    const std::string target_param_docstring{ fmt::format("create a CUDA {} with the provided target platform and parameters", csvm_name) };
    const std::string kwargs_docstring{ fmt::format("create a CUDA {} with the provided keyword arguments", csvm_name) };
    const std::string target_kwargs_docstring{ fmt::format("create a CUDA {} with the provided target platform and keyword arguments", csvm_name) };

    py::class_<backend_csvm_type, plssvm::cuda::csvm, csvm_type>(m, csvm_name.c_str(), class_docstring.c_str())
        .def(py::init([](const plssvm::parameter &params, plssvm::mpi::communicator comm) {
                 return std::make_unique<backend_csvm_type>(std::move(comm), params);
             }),
             param_docstring.c_str(),
             py::arg("params"),
             py::pos_only(),
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](const plssvm::target_platform target, const plssvm::parameter &params, plssvm::mpi::communicator comm) {
                 return std::make_unique<backend_csvm_type>(std::move(comm), target, params);
             }),
             target_param_docstring.c_str(),
             py::arg("target"),
             py::arg("params"),
             py::pos_only(),
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "comm, kernel_type", "degree", "gamma", "coef0", "cost" });
                 // create the MPI communicator
                 plssvm::mpi::communicator comm = args.contains("comm") ? args["comm"].cast<plssvm::mpi::communicator>() : plssvm::mpi::communicator{};
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(std::move(comm), params);
             }),
             kwargs_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "comm", "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // create the MPI communicator
                 plssvm::mpi::communicator comm = args.contains("comm") ? args["comm"].cast<plssvm::mpi::communicator>() : plssvm::mpi::communicator{};
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // create C-SVM with the provided target platform
                 return std::make_unique<backend_csvm_type>(std::move(comm), target, params);
             }),
             target_kwargs_docstring.c_str())
        .def("__repr__", [csvm_name](const backend_csvm_type &self) {
            return fmt::format("<plssvm.cuda.{} with {{ #devices: {} }}>", csvm_name, self.num_available_devices());
        });
}

}  // namespace

void init_cuda_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the CUDA C-SVM bindings
    py::module_ cuda_module = m.def_submodule("cuda", "a module containing all CUDA backend specific functionality");
    const py::module_ cuda_pure_virtual_module = cuda_module.def_submodule("__pure_virtual", "a module containing all pure-virtual CUDA backend specific functionality");

    // bind the pure-virtual base CUDA C-SVM
    [[maybe_unused]] const py::class_<plssvm::cuda::csvm, plssvm::csvm> virtual_base_cuda_csvm(cuda_pure_virtual_module, "__pure_virtual_cuda_base_CSVM");

    // bind the specific CUDA C-SVC and C-SVR classes
    bind_cuda_csvms<plssvm::csvc>(cuda_module, "CSVC");
    bind_cuda_csvms<plssvm::csvr>(cuda_module, "CSVR");

    // register CUDA backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::cuda::backend_exception>(cuda_module, "BackendError", base_exception);
}

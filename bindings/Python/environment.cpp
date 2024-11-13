/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/environment.hpp"

#include "plssvm/backend_types.hpp"  // plssvm::backend_type, plssvm::list_available_backends

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types: std::variant

#include <vector>  // std::vector

namespace py = pybind11;

void init_environment(py::module_ &m) {
    // use its own submodule for the environment related bindings
    py::module_ env_module = m.def_submodule("environment", "a module containing all environment initialization and finalization functionality");

    // bind free functions managing environment setup and teardown
    env_module.def("is_initialized", &plssvm::environment::is_initialized, "check whether the environments have already been initialized");
    env_module.def("is_finalized", &plssvm::environment::is_finalized, "check whether the environments have already been finalized");

    env_module.def("initialize", py::overload_cast<const std::vector<plssvm::backend_type> &>(&plssvm::environment::initialize), "initialize all requested backends, if available", py::arg("backends_to_init") = plssvm::list_available_backends());
    env_module.def("finalize", &plssvm::environment::finalize, "finalize all environments");

    // bind plssvm::environment::scope_guard
    py::class_<plssvm::environment::scope_guard>(env_module, "ScopeGuard")
        .def(py::init<>())
        .def(py::init<std::vector<plssvm::backend_type>>());
}

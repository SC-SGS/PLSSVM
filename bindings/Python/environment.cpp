/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/environment.hpp"

#include "plssvm/backend_types.hpp"  // plssvm::backend_type, plssvm::list_available_backends

#include "bindings/Python/utility.hpp"  // check_kwargs_for_correctness

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/pytypes.h"   // py::kwargs
#include "pybind11/stl.h"       // support for STL types: std::variant

#include <cstddef>  // std::size_t
#include <memory>   // std::make_unique
#include <vector>   // std::vector

namespace py = pybind11;

void init_environment(py::module_ &m) {
    // use its own submodule for the environment related bindings
    py::module_ env_module = m.def_submodule("environment", "a module containing all environment initialization and finalization functionality");

    // bind enum class
    py::enum_<plssvm::environment::status>(m, "Status")
        .value("UNINITIALIZED", plssvm::environment::status::uninitialized, "the backend environment hasn't been initialized or finalized yet")
        .value("INITIALIZED", plssvm::environment::status::initialized, "the backend environment has been initialized but not finalized yet")
        .value("FINALIZED", plssvm::environment::status::finalized, "the backend environment has already been initialized and finalized")
        .value("UNNECESSARY", plssvm::environment::status::unnecessary, "no backend environment initialization or finalization necessary");

    // bind free functions
    env_module.def("get_backend_status", &plssvm::environment::get_backend_status, "get the environment status for the provided backend");

    env_module.def("initialize", [](const py::kwargs &args) {
        // check for valid keys
        check_kwargs_for_correctness(args, { "backends" });
        if (args.contains("backends")) {
            plssvm::environment::initialize(args["backends"].cast<std::vector<plssvm::backend_type>>());
        } else {
            plssvm::environment::initialize();
        } }, "initialize all available backends or only the optionally provided once");
    env_module.def("initialize", [](std::vector<std::string> cmd_args, const py::kwargs &args) {
        std::vector<char *> cmd_args_ptr(cmd_args.size());
        for (std::size_t i = 0; i < cmd_args.size(); ++i) {
            cmd_args_ptr[i] = cmd_args[i].data();
        }
        // assemble command line arguments
        int argc = static_cast<int>(cmd_args_ptr.size());
        char **argv = cmd_args_ptr.data();

        // check for valid keys
        check_kwargs_for_correctness(args, { "backends" });
        if (args.contains("backends")) {
            plssvm::environment::initialize(argc, argv, args["backends"].cast<std::vector<plssvm::backend_type>>());
        } else {
            plssvm::environment::initialize(argc, argv);
        } }, "initialize all available backends or only the optionally provided once using the provided command line parameters");

    env_module.def("finalize", [](const py::kwargs &args) {
        // check for valid keys
        check_kwargs_for_correctness(args, { "backends" });
        if (args.contains("backends")) {
            plssvm::environment::finalize(args["backends"].cast<std::vector<plssvm::backend_type>>());
        } else {
            plssvm::environment::finalize();
        } }, "finalize all available backends or only the optionally provided once");

    // bind plssvm::environment::scope_guard
    py::class_<plssvm::environment::scope_guard>(env_module, "ScopeGuard")
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "backends" });
                 if (args.contains("backends")) {
                     return std::make_unique<plssvm::environment::scope_guard>(args["backends"].cast<std::vector<plssvm::backend_type>>());
                 } else {
                     return std::make_unique<plssvm::environment::scope_guard>();
                 }
             }),
             "create a new scope_guard and initialize all available backends or only the optionally provided once")
        .def(py::init([](std::vector<std::string> cmd_args, const py::kwargs &args) {
                 std::vector<char *> cmd_args_ptr(cmd_args.size());
                 for (std::size_t i = 0; i < cmd_args.size(); ++i) {
                     cmd_args_ptr[i] = cmd_args[i].data();
                 }
                 // assemble command line arguments
                 int argc = static_cast<int>(cmd_args_ptr.size());
                 char **argv = cmd_args_ptr.data();

                 // check for valid keys
                 check_kwargs_for_correctness(args, { "backends" });
                 if (args.contains("backends")) {
                     return std::make_unique<plssvm::environment::scope_guard>(argc, argv, args["backends"].cast<std::vector<plssvm::backend_type>>());
                 } else {
                     return std::make_unique<plssvm::environment::scope_guard>(argc, argv);
                 }
             }),
             "create a new scope_guard and initialize all available backends or only the optionally provided once using the provided command line parameters")
        .def("backends", &plssvm::environment::scope_guard::backends, "return all initialized backends");
}

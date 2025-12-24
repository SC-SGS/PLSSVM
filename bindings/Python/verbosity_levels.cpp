/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/verbosity_levels.hpp"  // plssvm::verbosity_level, bitwise operator overloads, plssvm::verbosity

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings
#include "bindings/Python/utility.hpp"       // plssvm::bindings::python::util::register_implicit_str_enum_conversion

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"  // py::module_, py::self, py::arg, py::enum_

namespace py = pybind11;

void init_verbosity_levels(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::verbosity_level> py_enum(m, "VerbosityLevel", "enum.Flag", "Enum class for all possible verbosity levels used in our own logging infrastructure.");
    py_enum
        .value("QUIET", plssvm::verbosity_level::quiet, "nothing is logged to the standard output to stdout")
        .value("LIBSVM", plssvm::verbosity_level::libsvm, "log the same messages as LIBSVM (used for better LIBSVM conformity) to stdout")
        .value("TIMING", plssvm::verbosity_level::timing, "log all messages related to timing information to stdout")
        .value("WARNING", plssvm::verbosity_level::warning, "log all messages related to warning to stderr")
        .value("FULL", plssvm::verbosity_level::full, "log all messages to stdout");

    // enable implicit conversion from string to enum
    plssvm::bindings::python::util::register_implicit_str_enum_conversion<plssvm::verbosity_level>(py_enum);

    // enable or disable verbose output
    m.def("quiet", []() { plssvm::verbosity = plssvm::verbosity_level::quiet; }, "no command line output is made during calls to PLSSVM functions");
    m.def("get_verbosity", []() { return plssvm::verbosity; }, "get the currently set verbosity level for all PLSSVM outputs to stdout");
    m.def("set_verbosity", [](const plssvm::verbosity_level verb) { plssvm::verbosity = verb; }, "set the verbosity level for all PLSSVM outputs to stdout", py::arg("verbosity"));
}

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/solver_types.hpp"  // plssvm::solver_type

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings
#include "bindings/Python/utility.hpp"       // plssvm::bindings::python::util::register_implicit_str_enum_conversion

#include "pybind11/pybind11.h"  // py::module_, py::enum_

namespace py = pybind11;

void init_solver_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::solver_type> py_enum(m, "SolverType", "enum.Enum", "Enum class for all possible solver types implemented in PLSSVM.");
    py_enum
        .value("AUTOMATIC", plssvm::solver_type::automatic, "the default solver type; depends on the available device and system memory")
        .value("CG_EXPLICIT", plssvm::solver_type::cg_explicit, "explicitly assemble the kernel matrix on the device")
        .value("CG_IMPLICIT", plssvm::solver_type::cg_implicit, "implicitly calculate the kernel matrix entries in each CG iteration");

    // enable implicit conversion from string to enum
    plssvm::bindings::python::util::register_implicit_str_enum_conversion<plssvm::solver_type>(py_enum);
}

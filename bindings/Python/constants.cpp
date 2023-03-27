/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/constants.hpp"

#include "pybind11/pybind11.h"  // py::module_

namespace py = pybind11;

void init_constants(py::module_ &m) {
    // enable or disable verbose output
    m.def(
        "quiet", [](const bool verb) { plssvm::verbose = !verb; }, "if set to true, no command line output is made during calls to PLSSVM functions");
}
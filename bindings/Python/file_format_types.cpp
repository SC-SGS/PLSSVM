/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/file_format_types.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_

namespace py = pybind11;

void init_file_format_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::file_format_type>(m, "FileFormatType")
        .value("LIBSVM", plssvm::file_format_type::libsvm, "the LIBSVM file format (default); for the file format specification see: https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html")
        .value("ARFF", plssvm::file_format_type::arff, "the ARFF file format; for the file format specification see: https://www.cs.waikato.ac.nz/~ml/weka/arff.html");
}
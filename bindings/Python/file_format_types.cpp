#include "plssvm/file_format_types.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_

namespace py = pybind11;

void init_file_format_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::file_format_type>(m, "FileFormatType")
        .value("LIBSVM", plssvm::file_format_type::libsvm)
        .value("ARFF", plssvm::file_format_type::arff);
}
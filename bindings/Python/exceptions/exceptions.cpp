#include "plssvm/exceptions/exceptions.hpp"

#include "utility.hpp"  // register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::exception

namespace py = pybind11;

void init_exceptions(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // register all basic PLSSVM exceptions
    register_py_exception<plssvm::invalid_parameter_exception>(m, "InvalidParameterError", base_exception);
    register_py_exception<plssvm::file_reader_exception>(m, "FileReaderError", base_exception);
    register_py_exception<plssvm::data_set_exception>(m, "DataSetError", base_exception);
    register_py_exception<plssvm::file_not_found_exception>(m, "FileNotFoundError", base_exception);
    register_py_exception<plssvm::invalid_file_format_exception>(m, "InvalidFileFormatError", base_exception);
    register_py_exception<plssvm::unsupported_backend_exception>(m, "UnsupportedBackendError", base_exception);
    register_py_exception<plssvm::unsupported_kernel_type_exception>(m, "UnsupportedKernelTypeError", base_exception);
    register_py_exception<plssvm::gpu_device_ptr_exception>(m, "GPUDevicePtrError", base_exception);
}
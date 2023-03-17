#include "plssvm/exceptions/exceptions.hpp"

#include "utility.hpp"  // PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_exceptions(py::module_ &m) {
    // register exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::invalid_parameter_exception, m, invalid_parameter_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::file_reader_exception, m, file_reader_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::data_set_exception, m, data_set_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::file_not_found_exception, m, file_not_found_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::invalid_file_format_exception, m, invalid_file_format_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::unsupported_backend_exception, m, unsupported_backend_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::unsupported_kernel_type_exception, m, unsupported_kernel_type_error)
    PLSSVM_REGISTER_EXCEPTION(plssvm::gpu_device_ptr_exception, m, gpu_device_ptr_error)
}
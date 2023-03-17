#include "plssvm/exceptions/exceptions.hpp"

#include "utility.hpp"  // PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module, py::exception, py::register_exception_translator
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_exceptions(py::module &m) {
    // register exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::invalid_parameter_exception, invalid_parameter_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::file_reader_exception, file_reader_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::data_set_exception, data_set_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::file_not_found_exception, file_not_found_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::invalid_file_format_exception, invalid_file_format_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::unsupported_backend_exception, unsupported_backend_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::unsupported_kernel_type_exception, unsupported_kernel_type_error);
    PLSSVM_REGISTER_EXCEPTION(plssvm::gpu_device_ptr_exception, gpu_device_ptr_error);
}
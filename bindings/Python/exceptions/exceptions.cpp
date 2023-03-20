#include "plssvm/exceptions/exceptions.hpp"

#include "utility.hpp"  // PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_

namespace py = pybind11;

void init_exceptions(py::module_ &m) {
    // register exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::invalid_parameter_exception, m, InvalidParameterError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::file_reader_exception, m, FileReaderError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::data_set_exception, m, DataSetError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::file_not_found_exception, m, FileNotFoundError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::invalid_file_format_exception, m, InvalidFileFormatError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::unsupported_backend_exception, m, UnsupportedBackendError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::unsupported_kernel_type_exception, m, UnsupportedKernelTypeError)
    PLSSVM_REGISTER_EXCEPTION(plssvm::gpu_device_ptr_exception, m, GPUDevicePtrError)
}
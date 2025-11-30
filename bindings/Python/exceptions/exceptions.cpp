/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/exceptions/exceptions.hpp"  // PLSSVM specific exceptions

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::register_py_exception
#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings

#include "pybind11/pybind11.h"  // py::module_, py::exception

namespace py = pybind11;

void init_exceptions(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // register all basic PLSSVM exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::invalid_parameter_exception>(m, "InvalidParameterError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::file_reader_exception>(m, "FileReaderError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::data_set_exception>(m, "DataSetError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::min_max_scaler_exception>(m, "MinMaxScalerError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::file_not_found_exception>(m, "FileNotFoundError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::invalid_file_format_exception>(m, "InvalidFileFormatError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::unsupported_backend_exception>(m, "UnsupportedBackendError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::unsupported_kernel_type_exception>(m, "UnsupportedKernelTypeError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::gpu_device_ptr_exception>(m, "GPUDevicePtrError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::matrix_exception>(m, "MatrixError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::kernel_launch_resources>(m, "KernelLaunchResourcesError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::classification_report_exception>(m, "ClassificationReportError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::regression_report_exception>(m, "RegressionReportError", base_exception);
    plssvm::bindings::python::util::register_py_exception<plssvm::environment_exception>(m, "EnvironmentError", base_exception);
}

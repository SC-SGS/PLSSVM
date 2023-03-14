#include "plssvm/exceptions/exceptions.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_exceptions(py::module &m) {
    // register exceptions
    static py::exception<plssvm::invalid_parameter_exception> invalid_parameter_exception_exc(m, "invalid_parameter_exception");
    static py::exception<plssvm::file_reader_exception> file_reader_exception_exc(m, "file_reader_exception");
    static py::exception<plssvm::data_set_exception> data_set_exception_exc(m, "data_set_exception");
    static py::exception<plssvm::file_not_found_exception> file_not_found_exception_exc(m, "file_not_found_exception");
    static py::exception<plssvm::invalid_file_format_exception> invalid_file_format_exception_exc(m, "invalid_file_format_exception");
    static py::exception<plssvm::unsupported_backend_exception> unsupported_backend_exception_exc(m, "unsupported_backend_exception");
    static py::exception<plssvm::unsupported_kernel_type_exception> unsupported_kernel_type_exception_exc(m, "unsupported_kernel_type_exception");
    static py::exception<plssvm::gpu_device_ptr_exception> gpu_device_ptr_exception_exc(m, "gpu_device_ptr_exception");

    // translate exceptions using the what_with_loc message instead of the what message
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const plssvm::invalid_parameter_exception &e) {
            invalid_parameter_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::file_reader_exception &e) {
            file_reader_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::data_set_exception &e) {
            data_set_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::file_not_found_exception &e) {
            file_not_found_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::invalid_file_format_exception &e) {
            invalid_file_format_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::unsupported_backend_exception &e) {
            unsupported_backend_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::unsupported_kernel_type_exception &e) {
            unsupported_kernel_type_exception_exc(e.what_with_loc().c_str());
        } catch (const plssvm::gpu_device_ptr_exception &e) {
            gpu_device_ptr_exception_exc(e.what_with_loc().c_str());
        }
    });
}
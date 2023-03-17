#include "pybind11/pybind11.h"  // PYBIND11_MODULE

namespace py = pybind11;

// forward declare binding functions
void init_constants(py::module &);
void init_target_platforms(py::module &);
void init_backend_types(py::module &);
void init_file_format_types(py::module &);
void init_kernel_function_types(py::module &);
void init_parameter(py::module &);
void init_model(py::module &);
void init_data_set(py::module &);
void init_version(py::module &);
void init_exceptions(py::module &);
void init_csvm(py::module &);
void init_openmp_csvm(py::module &);
void init_cuda_csvm(py::module &);
void init_hip_csvm(py::module &);
void init_opencl_csvm(py::module &);

PYBIND11_MODULE(plssvm, m) {
    // NOTE: the order matters. DON'T CHANGE IT!
    init_constants(m);
    init_target_platforms(m);
    init_backend_types(m);
    init_file_format_types(m);
    init_kernel_function_types(m);
    init_parameter(m);
    init_model(m);
    init_data_set(m);
    init_version(m);
    init_exceptions(m);
    init_csvm(m);
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    init_openmp_csvm(m);
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    init_cuda_csvm(m);
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    init_hip_csvm(m);
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    init_opencl_csvm(m);
#endif
}
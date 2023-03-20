#include "pybind11/pybind11.h"  // PYBIND11_MODULE, py::module_

namespace py = pybind11;

// forward declare binding functions
void init_constants(py::module_ &);
void init_target_platforms(py::module_ &);
void init_backend_types(py::module_ &);
void init_file_format_types(py::module_ &);
void init_kernel_function_types(py::module_ &);
void init_parameter(py::module_ &);
void init_model(py::module_ &);
void init_data_set(py::module_ &);
void init_version(py::module_ &);
void init_exceptions(py::module_ &);
void init_csvm(py::module_ &);
void init_openmp_csvm(py::module_ &);
void init_cuda_csvm(py::module_ &);
void init_hip_csvm(py::module_ &);
void init_opencl_csvm(py::module_ &);
void init_sycl(py::module_ &);
void init_hipsycl_csvm(py::module_ &);
void init_dpcpp_csvm(py::module_ &);

PYBIND11_MODULE(plssvm, m) {
    m.doc() = "Parallel Least Squares Support Vector Machine";

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

    // init bindings for the specific backends ONLY if the backend has been enabled
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
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    init_sycl(m);
    #if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    init_hipsycl_csvm(m);
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    init_dpcpp_csvm(m);
    #endif
#endif
}
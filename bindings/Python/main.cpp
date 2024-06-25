/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "pybind11/pybind11.h"  // PYBIND11_MODULE, py::module_, py::exception, py::register_exception_translator
#include "pybind11/pytypes.h"   // py::set_error

#include <exception>  // std::exception_ptr, std::rethrow_exception

namespace py = pybind11;

// forward declare binding functions
void init_verbosity_levels(py::module_ &);
void init_performance_tracker(py::module_ &);
void init_events(py::module_ &);
void init_hardware_sampler(py::module_ &, const py::exception<plssvm::exception> &);
void init_cpu_hardware_sampler(py::module_ &);
void init_gpu_nvidia_hardware_sampler(py::module_ &);
void init_gpu_amd_hardware_sampler(py::module_ &);
void init_target_platforms(py::module_ &);
void init_solver_types(py::module_ &);
void init_backend_types(py::module_ &);
void init_gamma(py::module_ &);
void init_classification_types(py::module_ &);
void init_file_format_types(py::module_ &);
void init_kernel_function_types(py::module_ &);
void init_kernel_functions(py::module_ &);
void init_parameter(py::module_ &);
void init_model(py::module_ &);
void init_data_set(py::module_ &);
void init_version(py::module_ &);
void init_exceptions(py::module_ &, const py::exception<plssvm::exception> &);
void init_csvm(py::module_ &);
void init_openmp_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_stdpar_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_cuda_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_hip_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_opencl_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_sycl(py::module_ &, const py::exception<plssvm::exception> &);
void init_sklearn(py::module_ &);

PYBIND11_MODULE(plssvm, m) {
    m.doc() = "Parallel Least Squares Support Vector Machine";

    // register PLSSVM base exception
    static py::exception<plssvm::exception> base_exception(m, "PLSSVMError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const plssvm::exception &e) {
            py::set_error(base_exception, e.what_with_loc().c_str());
        }
    });

    // NOTE: the order matters. DON'T CHANGE IT!
    init_verbosity_levels(m);

    // init performance tracking and hardware sampling bindings if the functionality has been enabled
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    init_performance_tracker(m);
    init_events(m);
    init_hardware_sampler(m, base_exception);
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    init_cpu_hardware_sampler(m);
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
    init_gpu_nvidia_hardware_sampler(m);
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
    init_gpu_amd_hardware_sampler(m);
#endif

    init_target_platforms(m);
    init_solver_types(m);
    init_backend_types(m);
    init_gamma(m);
    init_classification_types(m);
    init_file_format_types(m);
    init_kernel_function_types(m);
    init_kernel_functions(m);
    init_parameter(m);
    init_model(m);
    init_data_set(m);
    init_version(m);
    init_exceptions(m, base_exception);
    init_csvm(m);

    // init bindings for the specific backends ONLY if the backend has been enabled
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    init_openmp_csvm(m, base_exception);
#endif
#if defined(PLSSVM_HAS_STDPAR_BACKEND)
    init_stdpar_csvm(m, base_exception);
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    init_cuda_csvm(m, base_exception);
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    init_hip_csvm(m, base_exception);
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    init_opencl_csvm(m, base_exception);
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    init_sycl(m, base_exception);
#endif

    init_sklearn(m);
}

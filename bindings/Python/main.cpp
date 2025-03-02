/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/logging/mpi_log_untracked.hpp"  // plssvm::detail::log_untracked
#include "plssvm/environment.hpp"                       // plssvm::environment::{initialize, finalize}
#include "plssvm/exceptions/exceptions.hpp"             // plssvm::exception
#include "plssvm/mpi/communicator.hpp"                  // plssvm::mpi::communicator
#include "plssvm/mpi/environment.hpp"                   // plssvm::mpi::is_executed_via_mpirun
#include "plssvm/verbosity_levels.hpp"                  // plssvm::verbosity_level
#include "plssvm/version/version.hpp"                   // plssvm::version::version

#include "pybind11/pybind11.h"  // PYBIND11_MODULE, py::module_, py::exception, py::register_exception_translator
#include "pybind11/pytypes.h"   // py::set_error

#include <exception>  // std::exception_ptr, std::rethrow_exception

namespace py = pybind11;

// forward declare binding functions
void init_verbosity_levels(py::module_ &);
void init_performance_tracker(py::module_ &);
void init_events(py::module_ &);
void init_target_platforms(py::module_ &);
void init_solver_types(py::module_ &);
void init_svm_types(py::module_ &);
void init_backend_types(py::module_ &);
void init_gamma(py::module_ &);
void init_classification_types(py::module_ &);
void init_file_format_types(py::module_ &);
void init_kernel_function_types(py::module_ &);
void init_parameter(py::module_ &);
void init_kernel_functions(py::module_ &);
void init_classification_model(py::module_ &);
void init_regression_model(py::module_ &);
void init_min_max_scaler(py::module_ &);
void init_classification_data_set(py::module_ &);
void init_regression_data_set(py::module_ &);
void init_version(py::module_ &);
void init_exceptions(py::module_ &, const py::exception<plssvm::exception> &);
void init_regression_report(py::module_ &);
void init_csvm(py::module_ &);
void init_csvc(py::module_ &);
void init_csvr(py::module_ &);
void init_openmp_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_hpx_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_stdpar_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_cuda_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_hip_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_opencl_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_sycl(py::module_ &, const py::exception<plssvm::exception> &);
void init_kokkos_csvm(py::module_ &, const py::exception<plssvm::exception> &);
void init_sklearn_svc(py::module_ &);
void init_sklearn_svr(py::module_ &);

PYBIND11_MODULE(plssvm, m) {
    m.doc() = "PLSSVM - Parallel Least Squares Support Vector Machine";
    m.attr("__version__") = plssvm::version::version;

    // create a pure-virtual module
    py::module_ pure_virtual = m.def_submodule("__pure_virtual");

    // automatically initialize the environments
    plssvm::environment::initialize();

    // issue a warning if PLSSVM was build without MPI support, but the Python code was run via mpirun
#if !defined(PLSSVM_HAS_MPI_ENABLED)
    if (plssvm::mpi::is_executed_via_mpirun()) {
        plssvm::detail::log_untracked(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                      plssvm::mpi::communicator{},
                                      "WARNING: PLSSVM was built without MPI support, but is currently executed via mpirun! "
                                      "As a result, each MPI process will run the same code.\n");
    }
#endif

    // issue a warning if PLSSVM was build with MPI support and mpi4py wasn't found, but the Python code was still run via mpirun
#if defined(PLSSVM_HAS_MPI_ENABLED)
    if (plssvm::mpi::is_executed_via_mpirun()) {
        try {
            [[maybe_unused]] const py::module_ module = py::module_::import("mpi4py.MPI");
            // it worked
        } catch (const py::error_already_set &) {
            // error loading mpi4py -> issue the warning
            plssvm::detail::log_untracked(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                          plssvm::mpi::communicator{},
                                          "WARNING: PLSSVM was built without MPI support and the current code is executed via mpirun, but mpi4py wasn't found!\n");
        }
    }
#endif

    // automatically finalize the environments
    m.add_object("_cleanup", py::capsule([]() {
                     plssvm::environment::finalize();
                 }));

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
#endif

    init_target_platforms(m);
    init_solver_types(m);
    init_svm_types(m);
    init_backend_types(m);
    init_gamma(m);
    init_classification_types(m);
    init_file_format_types(m);
    init_kernel_function_types(m);
    init_parameter(m);
    init_kernel_functions(m);
    init_classification_model(m);
    init_regression_model(m);
    init_min_max_scaler(m);
    init_classification_data_set(m);
    init_regression_data_set(m);
    init_version(m);
    init_exceptions(m, base_exception);
    init_regression_report(m);
    init_csvm(m);
    init_csvc(m);
    init_csvr(m);

    // init bindings for the specific backends ONLY if the backend has been enabled
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    init_openmp_csvm(m, base_exception);
#endif
#if defined(PLSSVM_HAS_HPX_BACKEND)
    init_hpx_csvm(m, base_exception);
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
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    init_kokkos_csvm(m, base_exception);
#endif

    init_sklearn_svc(m);
    init_sklearn_svr(m);
}

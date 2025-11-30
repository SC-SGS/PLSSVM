/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Header forward declaring all helper functions used to create the Python bindings.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_BINDINGS_FWD_HPP_
#define PLSSVM_BINDINGS_PYTHON_BINDINGS_FWD_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exceptions

#include "pybind11/pybind11.h"  // py::module_, py::exception

namespace py = pybind11;

// forward declare binding functions
void init_verbosity_levels(py::module_ &m);
void init_performance_tracker(py::module_ &m);
void init_events(py::module_ &m);
void init_target_platforms(py::module_ &m);
void init_solver_types(py::module_ &m);
void init_svm_types(py::module_ &m);
void init_backend_types(py::module_ &m);
void init_gamma(py::module_ &m);
void init_classification_types(py::module_ &m);
void init_file_format_types(py::module_ &m);
void init_kernel_function_types(py::module_ &m);
void init_parameter(py::module_ &m);
void init_kernel_functions(py::module_ &m);
void init_classification_model(py::module_ &m);
void init_regression_model(py::module_ &m);
void init_min_max_scaler(py::module_ &m);
void init_classification_data_set(py::module_ &m);
void init_regression_data_set(py::module_ &m);
void init_exceptions(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_regression_report(py::module_ &m);
void init_csvm(py::module_ &m);
void init_csvc(py::module_ &m);
void init_csvr(py::module_ &m);
void init_openmp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_hpx_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_stdpar_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_cuda_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_hip_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_opencl_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_sycl(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
py::module_ init_adaptivecpp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
py::module_ init_dpcpp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_kokkos_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception);
void init_sklearn_svc(py::module_ &m);
void init_sklearn_svr(py::module_ &m);

#endif  // PLSSVM_BINDINGS_PYTHON_BINDINGS_FWD_HPP_

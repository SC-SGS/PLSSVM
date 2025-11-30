/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Core header including all other necessary headers.
 */

#ifndef PLSSVM_CORE_HPP_
#define PLSSVM_CORE_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                        // all supported backend types
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // the SYCL specific data parallel kernels
#include "plssvm/backends/SYCL/implementation_types.hpp"   // the SYCL implementation type
#include "plssvm/classification_report.hpp"                // reports different metrics (precision, recall, f1 score, and support) for the different classes after scoring
#include "plssvm/classification_types.hpp"                 // all supported multi-class classification strategies
#include "plssvm/constants.hpp"                            // verbosity flag und compile-time constants
#include "plssvm/csvm_factory.hpp"                         // a factory function to instantiate a C-SVM using a runtime backend; includes the available backend C-SVMs
#include "plssvm/data_set/classification_data_set.hpp"     // a classification data set used for training a C-SVC
#include "plssvm/data_set/min_max_scaler.hpp"              // a min-max scaler for the data sets
#include "plssvm/data_set/regression_data_set.hpp"         // a regression data set used for training a C-SVR
#include "plssvm/environment.hpp"                          // environment management functions and classes
#include "plssvm/exceptions/exceptions.hpp"                // exception hierarchy
#include "plssvm/file_format_types.hpp"                    // all supported file format types
#include "plssvm/gamma.hpp"                                // the types of the gamma parameter
#include "plssvm/kernel_function_types.hpp"                // all supported kernel function types
#include "plssvm/kernel_functions.hpp"                     // implementation of all supported kernel functions
#include "plssvm/matrix.hpp"                               // a custom matrix class
#include "plssvm/model/classification_model.hpp"           // the model as a result of training a C-SVC
#include "plssvm/model/regression_model.hpp"               // the model as a result of training a C-SVR
#include "plssvm/mpi/communicator.hpp"                     // PLSSVM MPI communicator wrapper
#include "plssvm/parameter.hpp"                            // the C-SVM parameter
#include "plssvm/regression_report.hpp"                    // reports different metrics (e.g., mean squared error or R^2 score) for the regression task after scoring
#include "plssvm/shape.hpp"                                // shape for a matrix or device pointer
#include "plssvm/solver_types.hpp"                         // all supported solver types (e.g., Conjugate Gradients with explicit, streaming, or implicit kernel matrix generation)
#include "plssvm/svm/csvc.hpp"                             // the base C-SVC every backend is inheriting from
#include "plssvm/svm/csvr.hpp"                             // the base C-SVR every backend is inheriting from
#include "plssvm/target_platforms.hpp"                     // all supported target platforms
#include "plssvm/verbosity_levels.hpp"                     // all supported verbosity levels
#include "plssvm/version/version.hpp"                      // version information

/// The main namespace containing all public API functions.
namespace plssvm { }

/// Namespace containing versioning information.
namespace plssvm::version { }

/// Namespace containing environment setup and teardown functionality.
namespace plssvm::environment { }

/// Namespace containing environment initialization and finalization functionality specific details. **Should not** directly be used by users.
namespace plssvm::environment::detail { }

/// Namespace containing Git versioning information.
namespace plssvm::version::git_metadata { }

/// Namespace containing implementation details. **Should not** directly be used by users.
namespace plssvm::detail { }

/// Namespace containing implementation details for the IO related functions. **Should not** directly be used by users.
namespace plssvm::detail::io { }

/// Namespace containing implementation details for the command line interface functionality. **Should not** directly be used by users.
namespace plssvm::detail::cmd { }

/// Namespace containing MPI wrapper functionality.
namespace plssvm::mpi { }

/// Namespace containing implementation details for our MPI wrapper functionality. **Should not** directly be used by users.
namespace plssvm::mpi::detail { }

/// Namespace containing implementation details for the performance tracking functionality. **Should not** directly be used by users.
namespace plssvm::detail::tracking { }

/// Namespace containing implementation details for the performance tracking functionality. **Should not** directly be used by users.
namespace plssvm::detail::tracking::impl { }

/// Namespace containing implementation details for the custom literals representing memory sizes. **Should not** directly be used by users.
namespace plssvm::detail::literals { }

/// Namespace containing operator overloads for [std::vector](https://en.cppreference.com/w/cpp/container/vector) and other mathematical functions on vectors.
namespace plssvm::operators { }

/// Namespace containing the C-SVM using the OpenMP backend.
namespace plssvm::openmp { }

/// Namespace containing OpenMP backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::openmp::detail { }

/// Namespace containing the C-SVM using the HPX backend.
namespace plssvm::hpx { }

/// Namespace containing HPX backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::hpx::detail { }

/// Namespace containing the C-SVM using the stdpar backend.
namespace plssvm::stdpar { }

/// Namespace containing stdpar backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::stdpar::detail { }

/// Namespace containing the C-SVM using the CUDA backend.
namespace plssvm::cuda { }

/// Namespace containing CUDA backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::cuda::detail { }

/// Namespace containing the C-SVM using the HIP backend.
namespace plssvm::hip { }

/// Namespace containing HIP backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::hip::detail { }

/// Namespace containing the C-SVM using the OpenCL backend.
namespace plssvm::opencl { }

/// Namespace containing OpenCL backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::opencl::detail { }

/// Namespace containing the C-SVM using the SYCL backend with DPC++ as SYCL implementation.
namespace plssvm::dpcpp { }

/// Namespace containing the C-SVM using the SYCL backend with DPC++ as SYCL implementation. **Should not** directly be used by users.
namespace plssvm::dpcpp::detail { }

/// Namespace containing the C-SVM using the SYCL backend with AdaptiveCpp as SYCL implementation.
namespace plssvm::adaptivecpp { }

/// Namespace containing the C-SVM using the SYCL backend with AdaptiveCpp as SYCL implementation. **Should not** directly be used by users.
namespace plssvm::adaptivecpp::detail { }

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation.
namespace plssvm::sycl { }

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. **Should not** directly be used by users.
namespace plssvm::sycl::detail { }

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. Specific to the basic data parallel kernel. **Should not** directly be used by users.
namespace plssvm::sycl::detail::basic { }

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. Specific to the work-group data parallel kernel. **Should not** directly be used by users.
namespace plssvm::sycl::detail::work_group { }

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. Specific to the hierarchical data parallel kernel. **Should not** directly be used by users.
namespace plssvm::sycl::detail::hierarchical { }

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. Specific to the scoped parallelism kernel. **Should not** directly be used by users.
namespace plssvm::sycl::detail::scoped { }

/// Namespace containing the C-SVM using the Kokkos backend.
namespace plssvm::kokkos { }

/// Namespace containing Kokkos backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::kokkos::detail { }

/// Namespace containing Kokkos backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::kokkos::detail::impl { }

#endif  // PLSSVM_CORE_HPP_

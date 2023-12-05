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

#include "plssvm/csvm.hpp"                                  // the base C-SVM every backend is inheriting from
#include "plssvm/csvm_factory.hpp"                          // a factory function to instantiate a C-SVM using a runtime backend; includes the available backend C-SVMs

#include "plssvm/classification_report.hpp"                 // reports different metrics (precision, recall, f1 score, and support) for the different classes after scoring

#include "plssvm/constants.hpp"                             // verbosity flag und compile-time constants
#include "plssvm/parameter.hpp"                             // the C-SVM parameter

#include "plssvm/matrix.hpp"                                // a custom matrix class
#include "plssvm/data_set.hpp"                              // a data set used for training a C-SVM
#include "plssvm/model.hpp"                                 // the model as a result of training a C-SVM

#include "plssvm/backend_types.hpp"                         // all supported backend types
#include "plssvm/classification_types.hpp"                  // all supported multi-class classification strategies
#include "plssvm/kernel_function_types.hpp"                 // all supported kernel function types
#include "plssvm/solver_types.hpp"                          // all supported solver types (e.g., Conjugate Gradients with explicit, streaming, or implicit kernel matrix generation)
#include "plssvm/target_platforms.hpp"                      // all supported target platforms
#include "plssvm/verbosity_levels.hpp"                      // all supported verbosity levels

#include "plssvm/backends/SYCL/implementation_type.hpp"     // the SYCL implementation type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // the SYCL specific kernel invocation typ

#include "plssvm/exceptions/exceptions.hpp"                 // exception hierarchy
#include "plssvm/version/version.hpp"                       // version information

/// The main namespace containing all public API functions.
namespace plssvm {}

/// Namespace containing versioning information.
namespace plssvm::version {}

/// Namespace containing implementation details. **Should not** directly be used by users.
namespace plssvm::detail {}
/// Namespace containing implementation details for the IO related functions. **Should not** directly be used by users.
namespace plssvm::detail::io {}
/// Namespace containing implementation details for the command line interface functionality. **Should not** directly be used by users.
namespace plssvm::detail::cmd {}

/// Namespace containing operator overloads for [std::vector](https://en.cppreference.com/w/cpp/container/vector) and other mathematical functions on vectors.
namespace plssvm::operators {}

/// Namespace containing the C-SVM using the OpenMP backend.
namespace plssvm::openmp {}

/// Namespace containing the C-SVM using the CUDA backend.
namespace plssvm::cuda {}
/// Namespace containing CUDA backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::cuda::detail {}

/// Namespace containing the C-SVM using the HIP backend.
namespace plssvm::hip {}
/// Namespace containing HIP backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::hip::detail {}

/// Namespace containing the C-SVM using the OpenCL backend.
namespace plssvm::opencl {}
/// Namespace containing OpenCL backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::opencl::detail {}

/// Namespace containing the C-SVM using the SYCL backend with DPC++ as SYCL implementation.
namespace plssvm::dpcpp {
using namespace plssvm::sycl;
}
/// Namespace containing the C-SVM using the SYCL backend with DPC++ as SYCL implementation. **Should not** directly be used by users.
namespace plssvm::dpcpp::detail {}

/// Namespace containing the C-SVM using the SYCL backend with AdaptiveCpp as SYCL implementation.
namespace plssvm::adaptivecpp {
using namespace plssvm::sycl;
}
/// Namespace containing the C-SVM using the SYCL backend with AdaptiveCpp as SYCL implementation. **Should not** directly be used by users.
namespace plssvm::adaptivecpp::detail {}

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation.
namespace plssvm::sycl {
#if defined(PLSSVM_HAS_SYCL_BACKEND)
using namespace plssvm::PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION;
#endif
}  // namespace plssvm::sycl
/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. **Should not** directly be used by users.
namespace plssvm::sycl::detail {}

#endif  // PLSSVM_CORE_HPP_
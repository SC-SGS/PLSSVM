/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Core header including all other necessary headers.
 */

#pragma once

#include "plssvm/csvm_factory.hpp"

#include "plssvm/backend_types.hpp"
#include "plssvm/csvm.hpp"
#include "plssvm/exceptions/exceptions.hpp"
#include "plssvm/kernel_types.hpp"
#include "plssvm/parameter.hpp"
#include "plssvm/version/version.hpp"

/// The main namespace containing all API functions.
namespace plssvm {}

/// Namespace containing versioning information.
namespace plssvm::version {}

/// Namespace containing implementation details. **Should not** directly be used by users.
namespace plssvm::detail {}

/// Namespace containing the C-SVM using the OpenMP backend.
namespace plssvm::openmp {}

/// Namespace containing the C-SVM using the CUDA backend.
namespace plssvm::cuda {}

/// Namespace containing CUDA backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::cuda::detail {}

/// Namespace containing the C-SVM using the OpenCL backend.
namespace plssvm::opencl {}
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

#include "plssvm/CSVM.hpp"
#include "plssvm/backend_types.hpp"
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

/// Namespace containing CUDA specific functions.
namespace plssvm::cuda {}

/// Namespace containing CUDA specific implementation details. **Should not** directly be used by users.
namespace plssvm::cuda::detail {}

/// Namespace containing OpenMP specific functions.
namespace plssvm::openmp {}
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

/// Namespace containing implementation details and **should not** be directly used by users.
namespace plssvm::detail {}

/// Namespace containing versioning information.
namespace plssvm::version {}
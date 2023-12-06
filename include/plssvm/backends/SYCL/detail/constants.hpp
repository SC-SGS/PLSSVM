/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Global compile-time constants specific to the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_CONSTANTS_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_CONSTANTS_HPP_
#pragma once

/**
 * @brief Macro identifying the used SYCL compiler as AdaptiveCpp.
 */
#define PLSSVM_SYCL_BACKEND_COMPILER_ADAPTIVECPP 1

/**
 * @brief Macro identifying the used SYCL compiler as [DPC++](https://github.com/intel/llvm).
 */
#define PLSSVM_SYCL_BACKEND_COMPILER_DPCPP 0

#if defined(PLSSVM_HAS_SYCL_BACKEND)
// define the default used SYCL implementation
namespace plssvm::sycl {
using namespace plssvm::PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION;
}
#else
// define dpcpp as default SYCL namespace if no SYCL backend is available (to prevent compiler errors)
namespace plssvm::dpcpp{ }  // make sure the namespace exists
namespace plssvm::sycl {
using namespace plssvm::dpcpp;
}
#endif

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_CONSTANTS_HPP_
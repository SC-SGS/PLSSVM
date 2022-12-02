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

#pragma once

/**
 * @brief Macro identifying the used SYCL compiler as [hipSYCL](https://github.com/illuhad/hipSYCL).
 */
#define PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL 1

/**
 * @brief Macro identifying the used SYCL compiler as [DPC++](https://github.com/intel/llvm).
 */
#define PLSSVM_SYCL_BACKEND_COMPILER_DPCPP 0
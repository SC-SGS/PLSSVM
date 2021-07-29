/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
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

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Core header including all other necessary headers.
 */

#pragma once

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include <plssvm/backends/OpenMP/OpenMP_CSVM.hpp>
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include <plssvm/backends/CUDA/CUDA_CSVM.hpp>
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include <plssvm/backends/OpenCL/OpenCL_CSVM.hpp>
#endif

#include <plssvm/CSVM.hpp>
#include <plssvm/exceptions.hpp>
#include <plssvm/kernel_types.hpp>
#include <plssvm/version/version.hpp>

/// The main namespace containing all API functions.
namespace plssvm {};

/// Namespace containing implementation details and **should not** be directly used by users.
namespace plssvm::detail {};

/// Namespace containing versioning information.
namespace plssvm::version {};
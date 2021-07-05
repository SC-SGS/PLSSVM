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
#include <plssvm/version.hpp>

/// The main namespace containing all API functions.
namespace plssvm {};

/// Utility namespace containing internal utility functions or classes.
namespace plssvm::util {};

/// Namespace containing versioning information.
namespace plssvm::version {};
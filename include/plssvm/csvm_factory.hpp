/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Factory functions for constructing a new C-SVM based on the provided command line arguments.
 */

#pragma once

#include "plssvm/backend_types.hpp"          // plssvm::backend
#include "plssvm/csvm.hpp"                   // plssvm::csvm
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"  // fmt::format

#include <memory>  // std::unique_ptr, std::make_unique

// only include requested/available backends
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/csvm.hpp"  // plssvm::cuda::csvm
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/csvm.hpp"  // plssvm::opencl::csvm
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    #include "plssvm/backends/SYCL/csvm.hpp"  // plssvm::sycl::csvm
#endif

namespace plssvm {
/**
 * @brief Construct a new C-SVM with the parameters given through @p params using the requested backend.
 * @tparam T the type of the data
 * @param[in] params struct encapsulating all possible parameters
 * @throws unsupported_backend_exception if the requested backend isn't available
 * @return [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr) to the constructed C-SVM
 */
template <typename T>
std::unique_ptr<csvm<T>> make_csvm(const parameter<T> &params) {
    switch (params.backend) {
        case backend_type::openmp:
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
            return std::make_unique<openmp::csvm<T>>(params);
#else
            throw unsupported_backend_exception{ "No OpenMP backend available!" };
#endif

        case backend_type::cuda:
#if defined(PLSSVM_HAS_CUDA_BACKEND)
            return std::make_unique<cuda::csvm<T>>(params);
#else
            throw unsupported_backend_exception{ "No CUDA backend available!" };
#endif

        case backend_type::opencl:
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
            return std::make_unique<opencl::csvm<T>>(params);
#else
            throw unsupported_backend_exception{ "No OpenCL backend available!" };
#endif
        case backend_type::sycl:
#if defined(PLSSVM_HAS_SYCL_BACKEND)
            return std::make_unique<sycl::csvm<T>>(params);
#else
            throw unsupported_backend_exception{ "No SYCL backend available!" };
#endif
    }
    throw unsupported_backend_exception{ fmt::format("Can't recognize backend with value '{}'!", detail::to_underlying(params.backend)) };
}

}  // namespace plssvm
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
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

#include <memory>   // std::unique_ptr, std::make_unique
#include <utility>  // std::forward

// only include requested/available backends
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/csvm.hpp"
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/csvm.hpp"
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/csvm.hpp"
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    #include "plssvm/backends/SYCL/csvm.hpp"
#endif

namespace plssvm {

/**
  * @brief Construct a new C-SVM with the parameters given by @p args using the requested backend @p type.
  * @tparam T the type of the data
  * @tparam Args the types of parameters used to construct the C-SVM
  * @param[in] type the used backend
  * @param[in] args the used parameters
  * @return [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr) to the constructed C-SVM
  */
template <typename T, typename... Args>
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
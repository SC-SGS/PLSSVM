/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Factory function for constructing a new C-SVM using one of the available backends based on the provided command line arguments.
 */

#pragma once

#include "plssvm/backend_types.hpp"          // plssvm::backend
#include "plssvm/csvm.hpp"                   // plssvm::csvm
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include <memory>  // std::unique_ptr, std::make_unique

// only include requested/available backends
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/csvm.hpp"  // plssvm::cuda::csvm
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    #include "plssvm/backends/HIP/csvm.hpp"  // plssvm::hip::csvm
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/csvm.hpp"  // plssvm::opencl::csvm
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
        #include "plssvm/backends/DPCPP/csvm.hpp"  // plssvm::dpcpp::csvm
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
        #include "plssvm/backends/hipSYCL/csvm.hpp"  // plssvm::hipsycl::csvm
    #endif
#endif

namespace plssvm {

/**
 * @brief Construct a new C-SVM with the parameters given through @p params using the requested backend.
 * @tparam T the type of the data
 * @param[in] params class encapsulating all possible parameters
 * @throws plssvm::unsupported_backend_exception if the requested backend isn't available
 * @return [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr) to the constructed C-SVM (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::unique_ptr<csvm<T>> make_csvm(const parameter<T> &params) {
    switch (params.backend) {
        case backend_type::automatic: {
            parameter<T> new_params{ params };
            new_params.backend = determine_default_backend();
            return make_csvm(new_params);
        }
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

        case backend_type::hip:
#if defined(PLSSVM_HAS_HIP_BACKEND)
            return std::make_unique<hip::csvm<T>>(params);
#else
            throw unsupported_backend_exception{ "No HIP backend available!" };
#endif

        case backend_type::opencl:
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
            return std::make_unique<opencl::csvm<T>>(params);
#else
            throw unsupported_backend_exception{ "No OpenCL backend available!" };
#endif

        case backend_type::sycl:
#if defined(PLSSVM_HAS_SYCL_BACKEND)
            switch (params.sycl_implementation_type) {
                case sycl::implementation_type::automatic:
                    return std::make_unique<PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION::csvm<T>>(params);
                case sycl::implementation_type::dpcpp:
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
                    return std::make_unique<dpcpp::csvm<T>>(params);
    #else
                    throw unsupported_backend_exception{ "No SYCL backend using DPC++ available!" };
    #endif
                case sycl::implementation_type::hipsycl:
    #if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
                    return std::make_unique<hipsycl::csvm<T>>(params);
    #else
                    throw unsupported_backend_exception{ "No SYCL backend using hipSYCL available!" };
    #endif
            }
#else
            throw unsupported_backend_exception{ "No SYCL backend available!" };
#endif
    }
    throw unsupported_backend_exception{ "Can't recognize backend !" };
}

}  // namespace plssvm

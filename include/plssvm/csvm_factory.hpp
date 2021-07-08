#pragma once

#include "plssvm/backend_types.hpp"          // plssvm::backend
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception

#include "fmt/core.h"  // fmt::format

#include <memory>   // std::unique_ptr, std::make_unique
#include <utility>  // std::forward

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/CUDA_CSVM.hpp"
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/OpenCL_CSVM.hpp"
#endif

namespace plssvm {

template <typename T, typename... Args>
std::unique_ptr<CSVM<T>> make_SVM(const backend_type type, Args... args) {
    switch (type) {
        case backend_type::openmp:
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
            return std::make_unique<OpenMP_CSVM<T>>(std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No OpenMP backend available!" };
#endif

        case backend_type::cuda:
#if defined(PLSSVM_HAS_CUDA_BACKEND)
            return std::make_unique<CUDA_CSVM>(std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No CUDA backend available!" };
#endif

        case backend_type::opencl:
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
            return std::make_unique<OpenCL_CSVM>(std::forward<Args>(args)...);
#else
            throw unsupported_backend_exception{ "No OpenCL backend available!" };
#endif
        default:
            throw unsupported_backend_exception{ fmt::format("Can't recognize backend with value '{}'!", static_cast<int>(type)) };
    }
}
}  // namespace plssvm
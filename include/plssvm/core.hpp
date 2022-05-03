/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Core header including all other necessary headers.
 */

#pragma once

#include "plssvm/csvm.hpp"
#include "plssvm/csvm_factory.hpp"

#include "plssvm/constants.hpp"
#include "plssvm/parameter.hpp"
#include "plssvm/parameter_train.hpp"
#include "plssvm/parameter_predict.hpp"

#include "plssvm/backend_types.hpp"
#include "plssvm/kernel_types.hpp"
#include "plssvm/target_platforms.hpp"

#include "plssvm/exceptions/exceptions.hpp"
#include "plssvm/version/version.hpp"

#include "plssvm/backends/SYCL/implementation_type.hpp"
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"


#include "plssvm/data_set.hpp"

/// The main namespace containing all public API functions.
namespace plssvm {}

/// Namespace containing versioning information.
namespace plssvm::version {}

/// Namespace containing implementation details. **Should not** directly be used by users.
namespace plssvm::detail {}

namespace plssvm::detail::io {}

/// Namespace containing operator overloads for [std::vector](https://en.cppreference.com/w/cpp/container/vector) and other mathematical functions on vectors.
namespace plssvm::operators {}

/// Namespace containing the C-SVM using the OpenMP backend.
namespace plssvm::openmp {}

/// Namespace containing the C-SVM using the CUDA backend.
namespace plssvm::cuda {}

/// Namespace containing CUDA backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::cuda::detail {}

/// Namespace containing the C-SVM using the HIP backend.
namespace plssvm::hip {}

/// Namespace containing HIP backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::hip::detail {}

/// Namespace containing the C-SVM using the OpenCL backend.
namespace plssvm::opencl {}

/// Namespace containing OpenCL backend specific implementation details. **Should not** directly be used by users.
namespace plssvm::opencl::detail {}

/// Namespace containing the C-SVM using the SYCL backend independent of the used SYCL implementation. **May not** directly be used by users. Use `plssvm::sycl` instead.
namespace plssvm::sycl_generic {}

/// Namespace containing SYCL backend specific implementation details independent of the used SYCL implementation. **Should not** directly be used by users.
namespace plssvm::sycl_generic::detail {}

/// Namespace containing the C-SVM using the SYCL backend with DPC++ as SYCL implementation.
namespace plssvm::dpcpp {
using namespace plssvm::sycl_generic;
}

/// Namespace containing the C-SVM using the SYCL backend with DPC++ as SYCL implementation. **Should not** directly be used by users.
namespace plssvm::dpcpp::detail {}

/// Namespace containing the C-SVM using the SYCL backend with hipSYCL as SYCL implementation.
namespace plssvm::hipsycl {
using namespace plssvm::sycl_generic;
}

/// Namespace containing the C-SVM using the SYCL backend with hipSYCL as SYCL implementation. **Should not** directly be used by users.
namespace plssvm::hipsycl::detail {}

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation.
namespace plssvm::sycl {
using namespace plssvm::sycl_generic;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
using namespace plssvm::PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION;
#endif
}

/// Namespace containing the C-SVM using the SYCL backend with the preferred SYCL implementation. **Should not** directly be used by users.
namespace plssvm::sycl::detail {}
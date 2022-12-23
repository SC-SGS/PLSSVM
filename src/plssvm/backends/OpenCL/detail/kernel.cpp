/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/kernel.hpp"

#include "plssvm/backends/OpenCL/detail/utility.hpp"  // PLSSVM_OPENCL_ERROR_CHECK

#include "CL/cl.h"  // cl_kernel, clReleaseKernel

#include <memory>   // std::addressof
#include <utility>  // std::exchange

namespace plssvm::opencl::detail {

kernel::kernel(cl_kernel compute_kernel_p) noexcept :
    compute_kernel{ compute_kernel_p } {}

kernel::kernel(kernel &&other) noexcept :
    compute_kernel{ std::exchange(other.compute_kernel, nullptr) } {}

kernel &kernel::operator=(kernel &&other) noexcept {
    if (this != std::addressof(other)) {
        compute_kernel = std::exchange(other.compute_kernel, nullptr);
    }
    return *this;
}

kernel::~kernel() {
    if (compute_kernel) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseKernel(compute_kernel), "error releasing cl_kernel");
    }
}

}  // namespace plssvm::opencl::detail
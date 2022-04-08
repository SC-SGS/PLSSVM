/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/kernel.hpp"

#include "CL/cl.h"  // cl_kernel, clReleaseKernel

#include <memory>   // std::addressof
#include <utility>  // std::exchange

namespace plssvm::opencl::detail {

kernel::kernel(cl_kernel p_compute_kernel) noexcept :
    compute_kernel{ p_compute_kernel } {}

kernel::kernel(kernel &&other) noexcept :
    compute_kernel{ std::exchange(other.compute_kernel, nullptr) } {}

kernel &kernel::operator=(kernel &&other) {
    if (this != std::addressof(other)) {
        compute_kernel = std::exchange(other.compute_kernel, nullptr);
    }
    return *this;
}

kernel::~kernel() {
    if (compute_kernel) {
        clReleaseKernel(compute_kernel);
    }
}

}  // namespace plssvm::opencl::detail
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a very small RAII wrapper around a cl_kernel.
 */

#pragma once

#include "CL/cl.h"  // cl_kernel, clReleaseKernel

namespace plssvm::opencl::detail {

/**
 * @brief RAII wrapper class around a cl_kernel.
 */
class kernel {
  public:
    /**
     * @brief Construct a new wrapper around the provided @p compute_kernel.
     * @param[in] p_compute_kernel the cl_kernel to wrap
     */
    explicit kernel(cl_kernel p_compute_kernel) noexcept :
        compute_kernel{ p_compute_kernel } {}

    /**
     * @brief Release the cl_kernel resources on destruction.
     */
    ~kernel() {
        if (compute_kernel) {
            clReleaseKernel(compute_kernel);
        }
    }

    /**
     * @brief Implicitly convert a kernel wrapper to an OpenCL cl_kernel.
     * @return the wrapped OpenCL cl_kernel
     */
    operator cl_kernel &() noexcept { return compute_kernel; }
    /**
     * @brief Implicitly convert a kernel wrapper to an OpenCL cl_kernel.
     * @return the wrapped OpenCL cl_kernel
     */
    operator const cl_kernel &() const noexcept { return compute_kernel; }

    /// The wrapped OpenCL cl_kernel.
    cl_kernel compute_kernel;
};

}  // namespace plssvm::opencl::detail
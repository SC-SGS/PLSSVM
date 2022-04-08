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

#include "CL/cl.h"  // cl_kernel

namespace plssvm::opencl::detail {

/**
 * @brief Enum class for all different OpenCL compute kernels.
 * @details Used to distinguish kernels in the `plssvm::opencl::detail::command_queue` class.
 */
enum class compute_kernel_name {
    /// The kernels to generate the `q` vector.
    q_kernel,
    /// The main C-SVM kernel.
    svm_kernel,
    /// The predict kernel for the linear kernel function.
    w_kernel,
    /// The predict kernels for the polynomial and rbf kernel functions.
    predict_kernel
};

/**
 * @brief RAII wrapper class around a cl_kernel.
 */
class kernel {
  public:
    /**
     * @brief Default construct empty kernel.
     */
    kernel() = default;
    /**
     * @brief Construct a new wrapper around the provided @p compute_kernel.
     * @param[in] compute_kernel the cl_kernel to wrap
     */
    explicit kernel(cl_kernel compute_kernel) noexcept;

    /**
     * @brief Delete copy-constructor to make #kernel a move only type.
     */
    kernel(const kernel &) = delete;
    /**
     * @brief Move-constructor as #kernel is a move-only type.
     * @param[in,out] other the kernel to move the resources from
     */
    kernel(kernel &&other) noexcept;
    /**
     * @brief Delete copy-assignment-operator to make #kernel a move only type.
     */
    kernel &operator=(const kernel &) = delete;
    /**
     * @brief Move-assignment-operator as #kernel is a move-only type.
     * @param[in,out] other the kernel to move the resources from
     * @return `*this`
     */
    kernel &operator=(kernel &&other);

    /**
     * @brief Release the cl_kernel resources on destruction.
     */
    ~kernel();

    /**
     * @brief Implicitly convert a kernel wrapper to an OpenCL cl_kernel.
     * @return the wrapped OpenCL cl_kernel (`[[nodiscard]]`)
     */
    [[nodiscard]] operator cl_kernel &() noexcept { return compute_kernel; }
    /**
     * @brief Implicitly convert a kernel wrapper to an OpenCL cl_kernel.
     * @return the wrapped OpenCL cl_kernel (`[[nodiscard]]`)
     */
    [[nodiscard]] operator const cl_kernel &() const noexcept { return compute_kernel; }

    /// The wrapped OpenCL cl_kernel.
    cl_kernel compute_kernel;
};

}  // namespace plssvm::opencl::detail
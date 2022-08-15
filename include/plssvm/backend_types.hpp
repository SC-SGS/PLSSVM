/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines all possible backends. Can also include backends not available on the current target platform.
 */

#ifndef PLSSVM_BACKEND_TYPES_HPP_
#define PLSSVM_BACKEND_TYPES_HPP_
#pragma once

#include "plssvm/target_platforms.hpp"  // plssvm::list_available_target_platforms

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm {

/**
 * @brief Enum class for all possible backend types.
 */
enum class backend_type {
    /** The default backend dependent on the specified target platforms. */
    automatic,
    /** [OpenMP](https://www.openmp.org/) to target CPUs only. */
    openmp,
    /** [CUDA](https://developer.nvidia.com/cuda-zone) to target NVIDIA GPUs only. */
    cuda,
    /** [HIP](https://github.com/ROCm-Developer-Tools/HIP) to target AMD and NVIDIA GPUs. */
    hip,
    /** [OpenCL](https://www.khronos.org/opencl/) to target GPUs from different vendors and CPUs. */
    opencl,
    /** [SYCL](https://www.khronos.org/sycl/) to target GPUs from different vendors and CPUs. Currently tested SYCL implementations are [DPC++](https://github.com/intel/llvm) and [hipSYCL](https://github.com/illuhad/hipSYCL). */
    sycl
};

/**
 * @brief Return a list of all currently available backends.
 * @details Only backends that where found during the CMake configuration are available.
 * @return the available backends (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<backend_type> list_available_backends();

/**
 * @brief Returns the default backend used given the specified target platforms during the CMake configuration.
 * @param[in] available_backends list of backends, if no backends are provided, queries all available backends
 * @param[in] available_target_platforms list of target platforms, if no target platforms are provided, queries all available target platforms
 * @return the default backend given the available backends and target platforms (`[[nodiscard]]`)
 */
[[nodiscard]] backend_type determine_default_backend(const std::vector<backend_type> &available_backends = list_available_backends(),
                                                     const std::vector<target_platform> &available_target_platforms = list_available_target_platforms());

/**
 * @brief Output the @p backend to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the backend type to
 * @param[in] backend the backend type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, backend_type backend);

/**
 * @brief Use the input-stream @p in to initialize the @p backend type.
 * @param[in,out] in input-stream to extract the backend type from
 * @param[in] backend the backend type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, backend_type &backend);

}  // namespace plssvm

#endif  // PLSSVM_BACKEND_TYPES_HPP_
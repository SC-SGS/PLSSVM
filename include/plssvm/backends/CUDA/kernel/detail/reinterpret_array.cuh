/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a utility function to reinterpret a 2D array inside a CUDA kernel.
 */

#ifndef PLSSVM_BACKENDS_CUDA_KERNEL_DETAIL_REINTERPRET_ARRAY_CUH_
#define PLSSVM_BACKENDS_CUDA_KERNEL_DETAIL_REINTERPRET_ARRAY_CUH_
#pragma once

#include <cstddef>  // std::size_t

namespace plssvm::cuda::detail {

/**
 * @brief Given the input array of dimensions [N][M], reinterpret it to be of dimensions [...][SIZE].
 * @tparam SIZE the new y-dimension size of the array
 * @tparam T the type of the values in the array
 * @tparam N the x-dimension of the array
 * @tparam M the y-dimension of the array
 * @param[in] input the array to reinterpret
 * @return the array of the newly specified dimensions (`[[nodiscard]]`)
 */
template <std::size_t SIZE, typename T, std::size_t N, std::size_t M>
[[nodiscard]] inline __device__ auto *reinterpret_array(T (&input)[N][M]) noexcept {
    return reinterpret_cast<T(*)[SIZE]>(input);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): reinterpret_cast necessary
}

/**
 * @brief Given the input array of dimensions [N], reinterpret it to be of dimensions [...][SIZE].
 * @tparam SIZE the new y-dimension size of the array
 * @tparam T the type of the values in the array
 * @tparam N the dimension of the array
 * @param[in] input the array to reinterpret
 * @return the array of the newly specified dimensions (`[[nodiscard]]`)
 */
template <std::size_t SIZE, typename T, std::size_t N>
[[nodiscard]] inline __device__ auto *reinterpret_array(T (&input)[N]) noexcept {
    return reinterpret_cast<T(*)[SIZE]>(input);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): reinterpret_cast necessary
}

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_KERNEL_DETAIL_REINTERPRET_ARRAY_CUH_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Function to list all available memory spaces at compile time.
 * @note Must be a separate file such that the Kokkos header must not be included in the "execution_space.hpp" file.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_CONSTEXPR_AVAILABLE_MEMORY_SPACES_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_CONSTEXPR_AVAILABLE_MEMORY_SPACES_HPP_

#include "plssvm/backends/Kokkos/memory_space.hpp"  // plssvm::kokkos::memory_space

#include <array>  // std::array

namespace plssvm::kokkos::detail {

/**
 * @brief List all available Kokkos::MemorySpaces at compile time.
 * @details The `memory_space::host_space` is always available!
 * @return a `std::array` containing all available memory spaces (`[[nodiscard]]`)
 */
[[nodiscard]] inline constexpr auto constexpr_available_memory_spaces() noexcept {
    // Note: the trailing comma is explicitly allowed by the standard
    return std::array{
        memory_space::host_space,
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_CUDA)
        memory_space::cuda_space,
        memory_space::cuda_usm_space,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_HIP)
        memory_space::hip_space,
        memory_space::hip_usm_space,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_SYCL)
        memory_space::sycl_space,
        memory_space::sycl_usm_space,
#endif
    };
}

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_CONSTEXPR_AVAILABLE_MEMORY_SPACES_HPP_

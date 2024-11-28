/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Function to list all available execution spaces at compile time.
 * @note Must be a separate file such that the Kokkos header must not be included in the "execution_space.hpp" file.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_CONSTEXPR_AVAILABLE_EXECUTION_SPACES_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_CONSTEXPR_AVAILABLE_EXECUTION_SPACES_HPP_

/**
 * @def PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES
 * @brief Set the macro `PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES` to 0 if it isn't already defined, i.e., no Kokkos execution space is available.
 *        Will normally be propagated by CMake with the number of available Kokkos execution spaces.
 */
#if !defined(PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES)
    #define PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES 0
#endif

#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space

#include <array>  // std::array

namespace plssvm::kokkos::detail {

/**
 * @brief List all available Kokkos::ExecutionSpaces at compile time.
 * @details At least one execution space must **always** be available!
 * @return a `std::array` containing all available execution spaces (`[[nodiscard]]`)
 */
[[nodiscard]] inline constexpr auto constexpr_available_execution_spaces() noexcept {
    // Note: The execution_space::automatic value may NEVER be added here!
    // Note: the trailing comma is explicitly allowed by the standard
    // Note: the order is intentionally chosen this way -> the order of the entries determines the priority when using a backend to run our code
    return std::array<execution_space, PLSSVM_KOKKOS_BACKEND_NUM_AVAILABLE_EXECUTION_SPACES>{
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_CUDA)
        execution_space::cuda,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_HIP)
        execution_space::hip,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_SYCL)
        execution_space::sycl,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_OPENMPTARGET)
        execution_space::openmp_target,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_OPENACC)
        execution_space::openacc,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_OPENMP)
        execution_space::openmp,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_THREADS)
        execution_space::threads,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_HPX)
        execution_space::hpx,
#endif
#if defined(PLSSVM_KOKKOS_BACKEND_ENABLE_SERIAL)
        execution_space::serial,
#endif
    };
}

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_CONSTEXPR_AVAILABLE_EXECUTION_SPACES_HPP_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Memory space type traits for the MemorySpaces in Kokkos.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_MEMORY_SPACE_TYPE_TRAITS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_MEMORY_SPACE_TYPE_TRAITS_HPP_
#include <decl/Kokkos_Declare_OPENMP.hpp>
#pragma once

#include "plssvm/backends/Kokkos/execution_space.hpp"              // plssvm::kokkos::execution_space
#include "plssvm/backends/Kokkos/execution_space_type_traits.hpp"  // plssvm::kokkos::kokkos_type_to_execution_space_v
#include "plssvm/backends/Kokkos/memory_space.hpp"                 // plssvm::kokkos::memory_space

#include "Kokkos_Core.hpp"  // Kokkos macros, Kokkos MemorySpace types

namespace plssvm::kokkos {

//***************************************************//
//            memory_space_to_kokkos_type            //
//***************************************************//

/**
 * @brief Uninstantiated base type to convert a `memory_space` enum value to a Kokkos::MemorySpace type.
 */
template <memory_space>
struct memory_space_to_kokkos_type;

/**
 * @brief Convert a `memory_space::host_space` enum value to a `Kokkos::HostSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::host_space> {
    using type = Kokkos::HostSpace;
};

#if defined(KOKKOS_ENABLE_CUDA)
/**
 * @brief Convert a `memory_space::cuda_space` enum value to a `Kokkos::CudaSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::cuda_space> {
    using type = Kokkos::CudaSpace;
};

/**
 * @brief Convert a `memory_space::cuda_usm_space` enum value to a `Kokkos::CudaUVMSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::cuda_usm_space> {
    using type = Kokkos::CudaUVMSpace;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
/**
 * @brief Convert a `memory_space::hip_space` enum value to a `Kokkos::HIPSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::hip_space> {
    using type = Kokkos::HIPSpace;
};

/**
 * @brief Convert a `memory_space::hip_usm_space` enum value to a `Kokkos::HIPManagedSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::hip_usm_space> {
    using type = Kokkos::HIPManagedSpace;
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
/**
 * @brief Convert a `memory_space::sycl_space` enum value to a `Kokkos::SYCLDeviceUSMSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::sycl_space> {
    using type = Kokkos::SYCLDeviceUSMSpace;
};

/**
 * @brief Convert a `memory_space::sycl_usm_space` enum value to a `Kokkos::SYCLSharedUSMSpace` Kokkos::MemorySpace type.
 */
template <>
struct memory_space_to_kokkos_type<memory_space::sycl_usm_space> {
    using type = Kokkos::SYCLSharedUSMSpace;
};
#endif

/**
 * @brief Convert the `memory_space` @p space to the corresponding Kokkos::MemorySpace type.
 * @tparam space the enum value to convert
 */
template <memory_space space>
using memory_space_to_kokkos_type_t = typename memory_space_to_kokkos_type<space>::type;

//***************************************************//
//            kokkos_type_to_memory_space            //
//***************************************************//

/**
 * @brief Uninstantiated base type to convert a Kokkos::MemorySpace type to a `memory_space` enum value.
 */
template <typename>
struct kokkos_type_to_memory_space;

/**
 * @brief Convert a `Kokkos::HostSpace` Kokkos::MemorySpace type to a `memory_space::host_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::HostSpace> {
    constexpr static memory_space value = memory_space::host_space;
};

#if defined(KOKKOS_ENABLE_CUDA)
/**
 * @brief Convert a `Kokkos::CudaSpace` Kokkos::MemorySpace type to a `memory_space::cuda_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::CudaSpace> {
    constexpr static memory_space value = memory_space::cuda_space;
};

/**
 * @brief Convert a `Kokkos::CudaUVMSpace` Kokkos::MemorySpace type to a `memory_space::cuda_usm_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::CudaUVMSpace> {
    constexpr static memory_space value = memory_space::cuda_usm_space;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
/**
 * @brief Convert a `Kokkos::HIPSpace` Kokkos::MemorySpace type to a `memory_space::hip_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::HIPSpace> {
    constexpr static memory_space value = memory_space::hip_space;
};

/**
 * @brief Convert a `Kokkos::HIPManagedSpace` Kokkos::MemorySpace type to a `memory_space::hip_usm_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::HIPManagedSpace> {
    constexpr static memory_space value = memory_space::hip_usm_space;
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
/**
 * @brief Convert a `Kokkos::SYCLDeviceUSMSpace` Kokkos::MemorySpace type to a `memory_space::sycl_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::SYCLDeviceUSMSpace> {
    constexpr static memory_space value = memory_space::sycl_space;
};

/**
 * @brief Convert a `Kokkos::SYCLSharedUSMSpace` Kokkos::MemorySpace type to a `memory_space::sycl_usm_space` enum value.
 */
template <>
struct kokkos_type_to_memory_space<Kokkos::SYCLSharedUSMSpace> {
    constexpr static memory_space value = memory_space::sycl_usm_space;
};
#endif

/**
 * @brief Convert the Kokkos::MemorySpace type @p MemorySpace to the corresponding `memory_space` enum value.
 * @tparam MemorySpace the Kokkos::MemorySpace type to convert
 */
template <typename MemorySpace>
inline constexpr memory_space kokkos_type_to_memory_space_v = kokkos_type_to_memory_space<MemorySpace>::value;

//***************************************************//
//          execution_space_to_memory_space          //
//***************************************************//

/**
 * @brief Convert a host `execution_space` enum value to a `memory_space::host_space` enum value.
 */
template <execution_space, bool UseUSM>
struct execution_space_to_memory_space {
    constexpr static memory_space value = memory_space::host_space;
};

/**
 * @brief Convert an `execution_space::cuda` that does not use USM allocations enum value to a `memory_space::cuda_space` enum value.
 */
template <>
struct execution_space_to_memory_space<execution_space::cuda, false> {
    constexpr static memory_space value = memory_space::cuda_space;
};

/**
 * @brief Convert an `execution_space::cuda` that does use USM allocations enum value to a `memory_space::cuda_usm_space` enum value.
 */
template <>
struct execution_space_to_memory_space<execution_space::cuda, true> {
    constexpr static memory_space value = memory_space::cuda_usm_space;
};

/**
 * @brief Convert an `execution_space::hip` that does not use USM allocations enum value to a `memory_space::hip_space` enum value.
 */
template <>
struct execution_space_to_memory_space<execution_space::hip, false> {
    constexpr static memory_space value = memory_space::hip_space;
};

/**
 * @brief Convert an `execution_space::hip` that does use USM allocations enum value to a `memory_space::hip_usm_space` enum value.
 */
template <>
struct execution_space_to_memory_space<execution_space::hip, true> {
    constexpr static memory_space value = memory_space::hip_usm_space;
};

/**
 * @brief Convert an `execution_space::sycl` that does not use USM allocations enum value to a `memory_space::sycl_space` enum value.
 */
template <>
struct execution_space_to_memory_space<execution_space::sycl, false> {
    constexpr static memory_space value = memory_space::sycl_space;
};

/**
 * @brief Convert an `execution_space::sycl` that does use USM allocations enum value to a `memory_space::sycl_usm_space` enum value.
 */
template <>
struct execution_space_to_memory_space<execution_space::sycl, true> {
    constexpr static memory_space value = memory_space::sycl_usm_space;
};

/**
 * @brief Convert the `execution_space` enum value @p space together with the @p UseUSM flag indication whether USM allocation should be used to the corresponding `memory_space` enum value.
 * @tparam space the `execution_space` enum value to convert
 * @tparam UseUSM `true` if USM allocations should be used
 */
template <execution_space space, bool UseUSM = false>
inline constexpr memory_space execution_space_to_memory_space_v = execution_space_to_memory_space<space, UseUSM>::value;

//***************************************************//
//   kokkos_execution_space_to_kokkos_memory_space   //
//***************************************************//

/**
 * @brief Convert the Kokkos::ExecutionSpace type together with the @p UseUSM flag indication whether USM allocation should be used to the corresponding Kokkos::MemorySpace type.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace type
 * @tparam UseUSM `true` if USM allocations should be used
 */
template <typename ExecutionSpace, bool UseUSM = false>
using kokkos_execution_space_to_kokkos_memory_space_t = memory_space_to_kokkos_type_t<execution_space_to_memory_space_v<kokkos_type_to_execution_space_v<ExecutionSpace>, UseUSM>>;

}  // namespace plssvm::kokkos

#endif  // PLSSVM_BACKENDS_KOKKOS_MEMORY_SPACE_TYPE_TRAITS_HPP_

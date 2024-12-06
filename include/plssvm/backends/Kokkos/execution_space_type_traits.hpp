/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Execution space type traits for the ExecutionSpaces in Kokkos.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_TYPE_TRAITS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_TYPE_TRAITS_HPP_
#pragma once

#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space

#include "Kokkos_Core.hpp"  // Kokkos macros, Kokkos ExecutionSpace types

namespace plssvm::kokkos {

//***************************************************//
//           execution_space_to_kokkos_type          //
//***************************************************//

/**
 * @brief Uninstantiated base type to convert an `execution_space` enum value to a Kokkos::ExecutionSpace type.
 */
template <execution_space>
struct execution_space_to_kokkos_type;

#if defined(KOKKOS_ENABLE_CUDA)
/**
 * @brief Convert an `execution_space::cuda` enum value to a `Kokkos::Cuda` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::cuda> {
    using type = Kokkos::Cuda;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
/**
 * @brief Convert an `execution_space::hip` enum value to a `Kokkos::HIP` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::hip> {
    using type = Kokkos::HIP;
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
/**
 * @brief Convert an `execution_space::sycl` enum value to a `Kokkos::SYCL` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::sycl> {
    using type = Kokkos::SYCL;
};
#endif

#if defined(KOKKOS_ENABLE_HPX)
/**
 * @brief Convert an `execution_space::hpx` enum value to a `Kokkos::Experimental::HPX` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::hpx> {
    using type = Kokkos::Experimental::HPX;
};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
/**
 * @brief Convert an `execution_space::openmp` enum value to a `Kokkos::OpenMP` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::openmp> {
    using type = Kokkos::OpenMP;
};
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
/**
 * @brief Convert an `execution_space::openmp_target` enum value to a `Kokkos::Experimental::OpenMPTarget` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::openmp_target> {
    using type = Kokkos::Experimental::OpenMPTarget;
};
#endif

#if defined(KOKKOS_ENABLE_OPENACC)
/**
 * @brief Convert an `execution_space::openacc` enum value to a `Kokkos::Experimental::OpenACC` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::openacc> {
    using type = Kokkos::Experimental::OpenACC;
};
#endif

#if defined(KOKKOS_ENABLE_THREADS)
/**
 * @brief Convert an `execution_space::threads` enum value to a `Kokkos::Threads` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::threads> {
    using type = Kokkos::Threads;
};
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
/**
 * @brief Convert an `execution_space::serial` enum value to a `Kokkos::Serial` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::serial> {
    using type = Kokkos::Serial;
};
#endif

/**
 * @brief Convert the `execution_space` @p space to the corresponding Kokkos::ExecutionSpace type.
 * @tparam space the enum value to convert
 */
template <execution_space space>
using execution_space_to_kokkos_type_t = typename execution_space_to_kokkos_type<space>::type;

//***************************************************//
//           kokkos_type_to_execution_space          //
//***************************************************//

/**
 * @brief Uninstantiated base type to convert a Kokkos::ExecutionSpace type to a `execution_space` enum value.
 */
template <typename>
struct kokkos_type_to_execution_space;

#if defined(KOKKOS_ENABLE_CUDA)
/**
 * @brief Convert a `Kokkos::Cuda` Kokkos::ExecutionSpace type to an `execution_space::cuda` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::Cuda> {
    constexpr static execution_space value = execution_space::cuda;
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
/**
 * @brief Convert a `Kokkos::HIP` Kokkos::ExecutionSpace type to an `execution_space::hip` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::HIP> {
    constexpr static execution_space value = execution_space::hip;
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
/**
 * @brief Convert a `Kokkos::SYCL` Kokkos::ExecutionSpace type to an `execution_space::sycl` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::SYCL> {
    constexpr static execution_space value = execution_space::sycl;
};
#endif

#if defined(KOKKOS_ENABLE_HPX)
/**
 * @brief Convert a `Kokkos::Experimental::HPX` Kokkos::ExecutionSpace type to an `execution_space::hpx` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::Experimental::HPX> {
    constexpr static execution_space value = execution_space::hpx;
};
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
/**
 * @brief Convert a `Kokkos::OpenMP` Kokkos::ExecutionSpace type to an `execution_space::openmp` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::OpenMP> {
    constexpr static execution_space value = execution_space::openmp;
};
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
/**
 * @brief Convert a `Kokkos::Experimental::OpenMPTarget` Kokkos::ExecutionSpace type to an `execution_space::openmp_target` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::Experimental::OpenMPTarget> {
    constexpr static execution_space value = execution_space::openmp_target;
};
#endif

#if defined(KOKKOS_ENABLE_OPENACC)
/**
 * @brief Convert a `Kokkos::Experimental::OpenACC` Kokkos::ExecutionSpace type to an `execution_space::openacc` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::Experimental::OpenACC> {
    constexpr static execution_space value = execution_space::openacc;
};
#endif

#if defined(KOKKOS_ENABLE_THREADS)
/**
 * @brief Convert a `Kokkos::Threads` Kokkos::ExecutionSpace type to an `execution_space::threads` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::Threads> {
    constexpr static execution_space value = execution_space::threads;
};
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
/**
 * @brief Convert a `Kokkos::Serial` Kokkos::ExecutionSpace type to an `execution_space::serial` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::Serial> {
    constexpr static execution_space value = execution_space::serial;
};
#endif

/**
 * @brief Convert the Kokkos::ExecutionSpace type @p ExecutionSpace to the corresponding `execution_space` enum value.
 * @tparam ExecutionSpace the Kokkos::ExecutionSpace type to convert
 */
template <typename ExecutionSpace>
inline constexpr execution_space kokkos_type_to_execution_space_v = kokkos_type_to_execution_space<ExecutionSpace>::value;

}  // namespace plssvm::kokkos

#endif  // PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_TYPE_TRAITS_HPP_

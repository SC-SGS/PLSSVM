/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Execution space enumeration for the ExecutionSpaces in Kokkos.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_HPP_
#define PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_HPP_
#pragma once

#include "Kokkos_Core.hpp"  // Kokkos macros, Kokkos ExecutionSpace types

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <array>   // std::array
#include <iosfwd>  // std::ostream forward declaration
#include <vector>  // std::vector

namespace plssvm::kokkos {

/**
 * @brief Enum class for all execution spaces supported by [Kokkos](https://github.com/kokkos/kokkos).
 */
enum class execution_space {
    /** Execution space representing execution on a CUDA device. */
    cuda,
    /** Execution space representing execution on a device supported by HIP. */
    hip,
    /** Execution space representing execution on a device supported by SYCL. */
    sycl,
    /** Execution space representing execution with the HPX runtime system. */
    hpx,
    /** Execution space representing execution with the OpenMP runtime system. */
    openmp,
    /** Execution space representing execution using the target offloading feature of the OpenMP runtime system. */
    openmp_target,
    /** Execution space representing execution with the OpenACC runtime system. */
    openacc,
    /** Execution space representing parallel execution with std::threads. */
    threads,
    /** Execution space representing serial execution on the CPU. Should always be available. */
    serial
};

/**
 * @brief Output the execution @p space to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the execution space to
 * @param[in] space the Kokkos execution space
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, execution_space space);

/**
 * @brief Use the input-stream @p in to initialize the execution @p space.
 * @param[in,out] in input-stream to extract the execution space from
 * @param[in] space the Kokkos execution space
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, execution_space &space);

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
 * @brief Convert an `execution_space::openmp_target` enum value to a `Kokkos::OpenMPTarget` Kokkos::ExecutionSpace type.
 */
template <>
struct execution_space_to_kokkos_type<execution_space::openmp_target> {
    using type = Kokkos::OpenMPTarget;
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
 * @brief Convert a `Kokkos::OpenMPTarget` Kokkos::ExecutionSpace type to an `execution_space::openmp_target` enum value.
 */
template <>
struct kokkos_type_to_execution_space<Kokkos::OpenMPTarget> {
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

//***************************************************//
//                  other functions                  //
//***************************************************//

namespace detail {

/**
 * @brief List all available Kokkos::ExecutionSpaces at compile time.
 * @details At least one execution space must **always** be available!
 * @return a `std::array` containing all available execution spaces (`[[nodiscard]]`)
 */
[[nodiscard]] inline constexpr auto constexpr_available_execution_spaces() noexcept {
    // Note: the trailing comma is explicitly allowed by the standard
    // Note: the order is intentionally chosen this way -> the order of the entries determines the priority when using a backend to run our code
    return std::array{
#if defined(KOKKOS_ENABLE_CUDA)
        execution_space::cuda,
#endif
#if defined(KOKKOS_ENABLE_HIP)
        execution_space::hip,
#endif
#if defined(KOKKOS_ENABLE_SYCL)
        execution_space::sycl,
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
        execution_space::openmp_target,
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
        execution_space::openacc,
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
        execution_space::openmp,
#endif
#if defined(KOKKOS_ENABLE_THREADS)
        execution_space::threads,
#endif
#if defined(KOKKOS_ENABLE_HPX)
        execution_space::hpx,
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
        execution_space::serial,
#endif
    };
}

}  // namespace detail

/**
 * @brief List all available Kokkos::ExecutionSpaces.
 * @details Only Kokkos::ExecutionSpaces that where enabled during the CMake configuration are available.
 * @return the available Kokkos::ExecutionSpaces (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<execution_space> list_available_execution_spaces();

}  // namespace plssvm::kokkos

/// @cond

template <>
struct fmt::formatter<plssvm::kokkos::execution_space> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_HPP_

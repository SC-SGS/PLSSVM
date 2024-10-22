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

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

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
    /** Execution space representing serial execution on the CPU. Always available. */
    serial
};

/**
 * @brief Create an `execution_space` from the current `Kokkos::DefaultExecutionSpace`.
 * @return the enum value representing the current `Kokkos::DefaultExecutionSpace` (`[[nodiscard]]`)
 */
[[nodiscard]] execution_space determine_execution_space() noexcept;

/**
 * @brief List all available Kokkos::ExecutionSpaces.
 * @details At least one execution space must **always** be available!
 * @return a vector containing all available execution spaces (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<execution_space> available_execution_spaces();

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

}  // namespace plssvm::kokkos

/// @endcond

template <>
struct fmt::formatter<plssvm::kokkos::execution_space> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_BACKENDS_KOKKOS_EXECUTION_SPACE_HPP_

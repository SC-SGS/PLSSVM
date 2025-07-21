/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Memory space enumeration for the MemorySpaces in Kokkos.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_MEMORY_SPACE_HPP_
#define PLSSVM_BACKENDS_KOKKOS_MEMORY_SPACE_HPP_
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>  // std::ostream forward declaration
#include <vector>  // std::vector

namespace plssvm::kokkos {

/**
 * @brief Enum class for all memory spaces supported by [Kokkos](https://github.com/kokkos/kokkos).
 */
enum class memory_space {
    /** Memory space representing traditional memory accessible from the CPU. */
    host_space,
    /** Memory space representing memory on a CUDA-capable GPU. */
    cuda_space,
    /** Memory space representing unified virtual memory on a CUDA-capable GPU system. */
    cuda_usm_space,
    /** Memory space representing memory in the HIP GPU programming environment. */
    hip_space,
    /** Memory space representing page-migrating memory in the HIP GPU programming environment. */
    hip_usm_space,
    /** Memory space representing device memory in the SYCL GPU programming environment. */
    sycl_space,
    /** Memory space representing page-migrating memory in the SYCL GPU programming environment */
    sycl_usm_space
};

/**
 * @brief Output the memory @p space to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the memory space to
 * @param[in] space the Kokkos memory space
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, memory_space space);

/**
 * @brief Use the input-stream @p in to initialize the memory @p space.
 * @param[in,out] in input-stream to extract the memory space from
 * @param[in] space the Kokkos memory space
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, memory_space &space);

/**
 * @brief List all available Kokkos::MemorySpaces.
 * @details Only Kokkos::MemorySpaces that where enabled during the CMake configuration are available.
 *          The `memory_space::host_space` is always included.
 * @return the available Kokkos::MemorySpaces (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<memory_space> list_available_memory_spaces();

}  // namespace plssvm::kokkos

/// @cond

template <>
struct fmt::formatter<plssvm::kokkos::memory_space> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_BACKENDS_KOKKOS_MEMORY_SPACE_HPP_

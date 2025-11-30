/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A simple struct encapsulating JIT compilation information.
 */

#ifndef PLSSVM_BACKENDS_OPENCL_DETAIL_JIT_INFO_HPP_
#define PLSSVM_BACKENDS_OPENCL_DETAIL_JIT_INFO_HPP_
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <chrono>  // std::chrono::milliseconds
#include <iosfwd>  // forward declare std::ostream and std::istream
#include <string>  // std::string

namespace plssvm::opencl::detail {

/**
 * @brief A struct encapsulating information regarding the current jit compilation.
 */
struct jit_info {
    /**
     * @brief An enumeration describing the state of the kernel cache.
     */
    enum class caching_status {
        /// The kernel cache was successful and could be used.
        success,
        /// No kernel cache for the current kernel versions found. JIT compile kernels again.
        error_no_cached_files,
        /// The number of cached files is wrong. JIT compile kernels again.
        error_invalid_number_of_cached_files,
    };

    /// `true` if inline PTX for the atomicAdd implementation on NVIDIA GPUs is used.
    bool use_ptx_inline{ false };
    /// The state of the kernel cache.
    caching_status cache_state{ caching_status::success };
    /// The kernel cache dir.
    std::string cache_dir;
    /// The duration of the JIT compilation.
    std::chrono::milliseconds duration{};
};

/**
 * @brief Output the @p status to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the JIT cache status type to
 * @param[in] status the JIT cache status
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, jit_info::caching_status status);

/**
 * @brief Create a JIT report from @p info to output if more than one MPI rank is active.
 * @param[in] info the JIT compilation information
 * @return the report string (`[[nodiscard]]`)
 */
[[nodiscard]] std::string create_jit_report(const jit_info &info);

}  // namespace plssvm::opencl::detail

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::opencl::detail::jit_info::caching_status> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_BACKENDS_OPENCL_DETAIL_JIT_INFO_HPP_

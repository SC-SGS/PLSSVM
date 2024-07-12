/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all supported stdpar implementations.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_IMPLEMENTATION_TYPE_HPP_
#define PLSSVM_BACKENDS_STDPAR_IMPLEMENTATION_TYPE_HPP_
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm::stdpar {

/**
 * @brief Enum class for all supported stdpar implementations.
 */
enum class implementation_type {
    /** Use [nvhpc (nvc++)](https://developer.nvidia.com/hpc-sdk) as stdpar implementation. */
    nvhpc,
    /** Use [roc-stdpar (patched LLVM)](https://github.com/ROCm/roc-stdpar) as stdpar implementation. */
    roc_stdpar,
    /** Use [icpx](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html) as stdpar implementation. */
    intel_llvm,
    /** Use [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (formerly known as hipSYCL) as stdpar implementation. */
    adaptivecpp,
    /** Use [GNU GCC + TBB](https://gcc.gnu.org/) as stdpar implementation */
    gnu_tbb
};

/**
 * @brief Return a list of all currently available stdpar implementations.
 * @details Currently **only** a single stdpar implementation can ever be active at a given time!.
 * @return the available stdpar implementations (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<implementation_type> list_available_stdpar_implementations();

/**
 * @brief Output the @p impl type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the stdpar implementation type to
 * @param[in] impl the stdpar implementation
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, implementation_type impl);

/**
 * @brief Use the input-stream @p in to initialize the @p impl type.
 * @param[in,out] in input-stream to extract the stdpar implementation type from
 * @param[in] impl the stdpar implementation
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, implementation_type &impl);

}  // namespace plssvm::stdpar

template <>
struct fmt::formatter<plssvm::stdpar::implementation_type> : fmt::ostream_formatter { };

#endif  // PLSSVM_BACKENDS_STDPAR_IMPLEMENTATION_TYPE_HPP_

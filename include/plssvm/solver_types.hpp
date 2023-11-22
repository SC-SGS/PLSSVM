/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible algorithm variations used to solve the LSSVM's system of linear equations.
 */

#ifndef PLSSVM_SOLVING_TYPES_HPP_
#define PLSSVM_SOLVING_TYPES_HPP_
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm {

/**
 * @brief Enum class for all possible solver types.
 */
enum class solver_type {
    /**
     * @brief The default solver.
     * @details 1. cg_explicit: if the kernel matrix can be fully stored in the available device memory (highest memory footprint, but fastest).
     *          2. cg_streaming: if the kernel matrix can be fully stored in the available host memory, but not in the device memory.
     *          3. cg_implicit: if the kernel matrix can neither be fully stored in host nor device memory (smallest memory footprint, but slowest).
     */
    automatic,
    /** Use the CG algorithm explicitly calculating the kernel matrix and fully storing it on the device. */
    cg_explicit,
    /** Use the CG algorithm explicitly calculating the kernel matrix, fully storing it on the host, and streaming the necessary parts to the device on the fly */
    cg_streaming,
    /** Use the CG algorithm implicitly recomputing the kernel matrix each CG iteration (smallest memory footprint). */
    cg_implicit
};

/**
 * @brief Output the @p solving type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the solving type to
 * @param[in] solving the solving type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, solver_type solving);

/**
 * @brief Use the input-stream @p in to initialize the @p solving type.
 * @param[in,out] in input-stream to extract the solving type from
 * @param[in] solving the solving type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, solver_type &solving);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::solver_type> : fmt::ostream_formatter {};

#endif  // PLSSVM_SOLVING_TYPES_HPP_

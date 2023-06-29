/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enum for the two different layout types Array-of-Structs and Struct-of-Arrays.
 */

#ifndef PLSSVM_DETAIL_LAYOUT_HPP_
#define PLSSVM_DETAIL_LAYOUT_HPP_
#pragma once

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // format types with a user defined operator<<

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm::detail {

/**
 * @brief Enum class for all available layout types.
 */
enum class layout_type {
    /** Array-of-Structs (AoS) */
    aos,
    /** Structs-of-Arrays (SoA) */
    soa
};

/**
 * @brief Output the @p layout to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the layout type to
 * @param[in] layout the layout type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, layout_type layout);

/**
 * @brief Use the input-stream @p in to initialize the @p layout type.
 * @param[in,out] in input-stream to extract the layout type from
 * @param[in] layout the layout type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, layout_type &layout);

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LAYOUT_HPP_
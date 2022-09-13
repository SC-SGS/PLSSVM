/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines functions to convert 2D vectors to 1D SoA or AoS vectors.
 */

#ifndef PLSSVM_DETAIL_LAYOUT_HPP_
#define PLSSVM_DETAIL_LAYOUT_HPP_
#pragma once

#include "plssvm/constants.hpp"      // plssvm::verbose
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT, PLSSVM_ASSERT_ENABLED

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // format types with a user defined operator<<

#include <algorithm>  // std::all_of
#include <chrono>     // std::chrono::{time_point, steady_clock, duration_cast}
#include <cstddef>    // std::size_t
#include <iosfwd>     // forward declare std::ostream and std::istream
#include <vector>     // std::vector

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
 * Output the @p layout to the given output-stream @p out.
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

/**
 * @brief Convert a 2D matrix into a 1D array in Array-of-Structs layout adding @p boundary_size values per data point for padding.
 * @details The 1D array is assembled data point wise. For more information see: https://www.wikiwand.com/en/AoS_and_SoA.
 * @tparam real_type the type of the values in the 2D matrix and the transformed 1D array
 * @param[in] matrix the 2D matrix to transform
 * @param[in] boundary_size the number of boundary elements to insert as padding
 * @param[in] num_points the number of data points to transform
 * @param[in] num_features the number of features per data point in the 2D matrix
 * @return the transformed 1D array in Array-of-Structs layout (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] inline std::vector<real_type> transform_to_aos_layout(const std::vector<std::vector<real_type>> &matrix, const std::size_t boundary_size, const std::size_t num_points, const std::size_t num_features) {
    std::vector<real_type> vec(num_points * (num_features + boundary_size));

#pragma omp parallel for collapse(2) default(none) shared(vec, matrix) firstprivate(num_features, num_points, boundary_size)
    for (std::size_t row = 0; row < num_points; ++row) {
        for (std::size_t col = 0; col < num_features; ++col) {
            vec[row * (num_features + boundary_size) + col] = matrix[row][col];
        }
    }

    return vec;
}

/**
 * @brief Convert a 2D matrix into a 1D array in Struct-of-Arrays layout adding @p boundary_size values per feature for padding.
 * @details The 1D array is assembled feature wise. For more information see: https://www.wikiwand.com/en/AoS_and_SoA.
 * @tparam real_type the type of the values in the 2D matrix and the transformed 1D array
 * @param[in] matrix the 2D matrix to transform
 * @param[in] boundary_size the number of boundary elements to insert as padding
 * @param[in] num_points the number of data points to transform
 * @param[in] num_features the number of features per data point in the 2D matrix
 * @return the transformed 1D array in Struct-of-Arrays layout (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] inline std::vector<real_type> transform_to_soa_layout(const std::vector<std::vector<real_type>> &matrix, const std::size_t boundary_size, const std::size_t num_points, const std::size_t num_features) {
    std::vector<real_type> vec(num_features * (num_points + boundary_size));

#pragma omp parallel for collapse(2) default(none) shared(vec, matrix) firstprivate(num_features, num_points, boundary_size)
    for (std::size_t col = 0; col < num_features; ++col) {
        for (std::size_t row = 0; row < num_points; ++row) {
            vec[col * (num_points + boundary_size) + row] = matrix[row][col];
        }
    }

    return vec;
}

/**
 * @brief Convert a 2D matrix into a 1D array in the @p layout adding @p boundary_size values per data point or feature respectively for padding.
 * @details For more information regarding AoS and SoA see: https://www.wikiwand.com/en/AoS_and_SoA.
 * @tparam real_type the type of the values in the 2D matrix and the transformed 1D array
 * @param[in] layout the layout type to transform the 2D matrix to, either Array-of-Structs (AoS) or Struct-of-Arrays (SoA)
 * @param[in] matrix the 2D matrix to transform
 * @param[in] boundary_size the number of boundary elements to insert as padding
 * @param[in] num_points the number of data points to transform
 * @return the transformed 1D array in the specified @p layout (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] inline std::vector<real_type> transform_to_layout(const layout_type layout, const std::vector<std::vector<real_type>> &matrix, const std::size_t boundary_size, const std::size_t num_points) {
    // perform some sanity checks
    PLSSVM_ASSERT(!matrix.empty(), "Matrix is empty!");
    PLSSVM_ASSERT(num_points <= matrix.size(), "Number of data points to transform can not exceed matrix size!");
    const typename std::vector<real_type>::size_type num_features = matrix.front().size();
#if defined(PLSSVM_ASSERT_ENABLED)
    const bool has_same_num_features = std::all_of(matrix.begin(), matrix.end(), [=](const std::vector<real_type> &point) { return point.size() == num_features; });
    PLSSVM_ASSERT(has_same_num_features, "Feature sizes mismatch! All features should have size {}.", num_features);
    PLSSVM_ASSERT(num_features > 0, "All features are empty!");
#endif

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    std::vector<real_type> ret;

    switch (layout) {
        case layout_type::aos:
            ret = transform_to_aos_layout(matrix, boundary_size, num_points, num_features);
            break;
        case layout_type::soa:
            ret = transform_to_soa_layout(matrix, boundary_size, num_points, num_features);
            break;
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Transformed dataset from 2D to 1D {} in {}.\n", layout, std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    return ret;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_LAYOUT_HPP_
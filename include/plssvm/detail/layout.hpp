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

#pragma once

#include "plssvm/constants.hpp"
#include "plssvm/detail/assert.hpp"

#include "fmt/core.h"
#include "fmt/ostream.h"

#include <algorithm>
#include <chrono>
#include <ostream>
#include <vector>

namespace plssvm::detail {

// TODO: test

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

enum class layout_type {
    aos,
    soa
};

std::ostream &operator<<(std::ostream &out, const layout_type layout) {
    switch (layout) {
        case layout_type::aos:
            return out << "Array-of-Structs (AoS)";
        case layout_type::soa:
            return out << "Struct-of-Arrays (SoA)";
    }
    return out << "unknown";
}

template <typename real_type>
[[nodiscard]] inline std::vector<real_type> transform_to_layout(const layout_type layout, const std::vector<std::vector<real_type>> &matrix, const std::size_t boundary_size, const std::size_t num_points) {
    // perform some sanity checks
    PLSSVM_ASSERT(!matrix.empty(), "Matrix is empty!");
    PLSSVM_ASSERT(num_points <= matrix.size(), "Num points to transform can not exceed matrix size!");
    const typename std::vector<real_type>::size_type num_features = matrix.front().size();
    PLSSVM_ASSERT(std::all_of(matrix.begin(), matrix.end(), [=](const std::vector<real_type> &point) { return point.size() == num_features; }), "Feature sizes mismatch! All features should have size {}!", num_features);

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

}
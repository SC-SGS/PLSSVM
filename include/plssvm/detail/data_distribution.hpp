/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements functions regarding the distribution of the data across multiple devices.
 */

#ifndef PLSSVM_DETAIL_DATA_DISTRIBUTION_HPP_
#define PLSSVM_DETAIL_DATA_DISTRIBUTION_HPP_
#pragma once

#include "plssvm/detail/memory_size.hpp"  // plssvm:detail::memory_size

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::detail {

/**
 * @brief Calculate the data distribution (i.e., the number of rows in the kernel matrix a *place* is responsible for) such that each *place* has
 *        approximately the same number of data points it is responsible for accounting only for the upper triangular matrix.
 * @details Example: if we have 10 data points, the number of entries in the triangular matrix is equal to 10 * (10 + 1) / 2 = 55.
 *          If we want to distribute these 10 data points across 2 devices, each device would be responsible for the following rows/data points:
 *          - device 0: rows 0, 1, and 2 -> 10 + 9 + 8 = **27 matrix entries**
 *          - device 1: rows 3, 4, 5, 6, 7, 8, 9 -> 7 + 6 + 5 + 4 + 3 + 2 + 1 = **28 matrix entries**
 *          Therefore, each device is responsible for approximately the same number of **matrix entries** and **not** the same number of **rows**!
 *
 *             num_data_points
 *              ____________
 *          0   \          |
 *          1    \     27  |  device 0
 *          2     \________|  ______
 *          3      \       |
 *          4       \  28  |
 *          5        \     |  device 1
 *          6         \    |
 *          7          \   |
 *          8           \  |
 *          9            \_|  ______
 *
 * @param[in] num_rows_reduced the number of data points to distribute
 * @param[in] num_places the number of places, i.e., different devices to distribute the data to
 * @return the distribution, e.g., device 0 is responsible for all kernel matrix rows starting from res[0] up-to res[1] (`[[nodiscard]]``)
 */
[[nodiscard]] std::vector<std::size_t> calculate_data_distribution(std::size_t num_rows_reduced, std::size_t num_places);

/**
 * @brief Given the @p data_distribution, returns the number of rows in the kernel matrix the @p place is responsible for.
 * @param[in] place the place to query
 * @param[in] data_distribution the previously determined data distribution
 * @return the number of place specific rows (`[[nodiscard]]`)
 */
[[nodiscard]] std::size_t get_place_specific_num_rows(std::size_t place, const std::vector<std::size_t> &data_distribution);

/**
 * @brief Given the @p data_distribution, returns the row offset, i.e., the first row @p place is responsible for.
 * @param[in] place the place to query
 * @param[in] data_distribution the previously determined data distribution
 * @return the first row the @p place is responsible for (`[[nodiscard]]`)
 */
[[nodiscard]] std::size_t get_place_row_offset(std::size_t place, const std::vector<std::size_t> &data_distribution);

/**
 * @brief Calculate the number of entries in the explicit kernel matrix for the @p place.
 * @param num_rows_reduced the total number of data points used in the kernel matrix construction (**with** dimensional reduction)
 * @param place the place to query
 * @param data_distribution the previously determined data distribution
 * @return the total number of entries, i.e., values to allocate on the @p place (`[[nodiscard]]`)
 */
[[nodiscard]] std::size_t calculate_explicit_kernel_matrix_num_entries_padded(std::size_t num_rows_reduced, std::size_t place, const std::vector<std::size_t> &data_distribution);

/**
 * @brief Calculate the theoretical total memory needed per place for explicitly assembling the kernel matrix.
 * @param[in] num_data_points the total number of data points (**without** dimensional reduction)
 * @param[in] num_features the total number of features
 * @param[in] num_classes the total number of classes
 * @param[in] data_distribution the previously determined data distribution
 * @return the theoretical total memory needed per place for cg_explicit (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<memory_size> calculate_maximum_explicit_kernel_matrix_memory_needed_per_place(std::size_t num_data_points, std::size_t num_features, std::size_t num_classes, const std::vector<std::size_t> &data_distribution);

/**
 * @brief Calculate the theoretical maximum single memory allocation size per place for explicitly assembling the kernel matrix.
 * @param[in] num_data_points the total number of data points (**without** dimensional reduction)
 * @param[in] num_features the total number of features
 * @param[in] num_classes the total number of classes
 * @param[in] data_distribution the previously determined data distribution
 * @return the theoretical maximum single memory allocation size per place for cg_explicit (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<memory_size> calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place(std::size_t num_data_points, std::size_t num_features, std::size_t num_classes, const std::vector<std::size_t> &data_distribution);

/**
 * @brief Calculate the theoretical total memory needed per place for implicitly assembling the kernel matrix.
 * @param[in] num_data_points the total number of data points (**without** dimensional reduction)
 * @param[in] num_features the total number of features
 * @param[in] num_classes the total number of classes
 * @param[in] data_distribution the previously determined data distribution
 * @return the theoretical total memory needed per place for cg_implicit (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<memory_size> calculate_maximum_implicit_kernel_matrix_memory_needed_per_place(std::size_t num_data_points, std::size_t num_features, std::size_t num_classes, const std::vector<std::size_t> &data_distribution);

/**
 * @brief Calculate the theoretical maximum single memory allocation size per place for implicitly assembling the kernel matrix.
 * @param[in] num_data_points the total number of data points (**without** dimensional reduction)
 * @param[in] num_features the total number of features
 * @param[in] num_classes the total number of classes
 * @param[in] data_distribution the previously determined data distribution
 * @return the theoretical maximum single memory allocation size per place for cg_implicit (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<memory_size> calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place(std::size_t num_data_points, std::size_t num_features, std::size_t num_classes, const std::vector<std::size_t> &data_distribution);

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_DATA_DISTRIBUTION_HPP_

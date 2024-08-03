/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements classes regarding different data distribution strategies.
 */

#ifndef PLSSVM_DETAIL_DATA_DISTRIBUTION_HPP_
#define PLSSVM_DETAIL_DATA_DISTRIBUTION_HPP_
#pragma once

#include "plssvm/detail/memory_size.hpp"  // plssvm:detail::memory_size

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstddef>  // std::size_t
#include <iosfwd>   // std::ostream forward declaration
#include <vector>   // std::vector

namespace plssvm::detail {

/**
 * @brief A data distribution for a specific number of rows distributed across a specific number of places (e.g., devices).
 */
class data_distribution {
  public:
    /**
     * @brief Default copy-constructor. Necessary due to the defaulted pure virtual destructor.
     */
    data_distribution(const data_distribution &) = default;
    /**
     * @brief Default move-constructor. Necessary due to the defaulted pure virtual destructor.
     */
    data_distribution(data_distribution &&) noexcept = default;
    /**
     * @brief Default copy-assignment operator. Necessary due to the defaulted pure virtual destructor.
     * @return `*this`
     */
    data_distribution &operator=(const data_distribution &) = default;
    /**
     * @brief Default move-assignment operator. Necessary due to the defaulted pure virtual destructor.
     * @return `*this`
     */
    data_distribution &operator=(data_distribution &&) noexcept = default;
    /**
     * @brief Default pure virtual destructor.
     */
    virtual ~data_distribution() = 0;

    /**
     * @brief Return the number of rows the @p place is responsible for.
     * @param[in] place the place to query
     * @return the number of place specific rows (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::size_t place_specific_num_rows(std::size_t place) const noexcept;
    /**
     * @brief Return the first row the @p place is responsible for.
     * @param[in] place the place to query
     * @return the first row the @p place is responsible for (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::size_t place_row_offset(std::size_t place) const noexcept;

    /**
     * @brief The data distribution of @p num_rows() values across @p num_places() places.
     * @return the data distribution (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::size_t> &distribution() const noexcept;
    /**
     * @brief The number of rows that have been distributed.
     * @return the number of rows (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_rows() const noexcept;
    /**
     * @brief The number of places (e.g., devices) to which the rows has been distributed.
     * @return the number of places (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_places() const noexcept;

  protected:
    /**
     * @brief Construct a new data distribution being able to hold the distribution for @p num_places.
     * @param[in] num_rows the number of rows to distribute (used to initialize the values in the distribution)
     * @param[in] num_places the number of places to distribute the rows to
     */
    data_distribution(std::size_t num_rows, std::size_t num_places);

    /// The specific data distribution across the requested number of places.
    std::vector<std::size_t> distribution_;
    /// The number of rows distributed.
    std::size_t num_rows_;
    /// The number of places the rows should be distributed to.
    std::size_t num_places_;
};

/**
 * @brief Output the @p dist to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the data distribution to
 * @param[in] dist the data distribution
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const data_distribution &dist);

//*************************************************************************************************************************************//
//                                                     specific data distributions                                                     //
//*************************************************************************************************************************************//

/**
 * @brief A triangular data distribution where each place is responsible for approximately the same number of values in a upper triangular matrix.
 */
class triangular_data_distribution : public data_distribution {
  public:
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
     * @param[in] num_rows the number of data points to distribute
     * @param[in] num_places the number of places, i.e., different devices to distribute the data to
     */
    triangular_data_distribution(std::size_t num_rows, std::size_t num_places);

    /**
     * @brief Calculate the number of entries in the explicit kernel matrix for the current number of rows and @p place.
     * @param[in] place the place (e.g., device) for which the number of matrix entries should be calculated
     * @return the total number of entries, i.e., values to allocate on the @p place (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t calculate_explicit_kernel_matrix_num_entries_padded(std::size_t place) const noexcept;

    /**
     * @brief Calculate the theoretical total memory needed per place for explicitly assembling the kernel matrix.
     * @param[in] num_features the total number of features
     * @param[in] num_classes the total number of classes
     * @return the theoretical total memory needed per place for cg_explicit (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<memory_size> calculate_maximum_explicit_kernel_matrix_memory_needed_per_place(std::size_t num_features, std::size_t num_classes) const;

    /**
     * @brief Calculate the theoretical maximum single memory allocation size per place for explicitly assembling the kernel matrix.
     * @param[in] num_features the total number of features
     * @param[in] num_classes the total number of classes
     * @return the theoretical maximum single memory allocation size per place for cg_explicit (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<memory_size> calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place(std::size_t num_features, std::size_t num_classes) const;

    /**
     * @brief Calculate the theoretical total memory needed per place for implicitly assembling the kernel matrix.
     * @param[in] num_features the total number of features
     * @param[in] num_classes the total number of classes
     * @return the theoretical total memory needed per place for cg_implicit (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<memory_size> calculate_maximum_implicit_kernel_matrix_memory_needed_per_place(std::size_t num_features, std::size_t num_classes) const;

    /**
     * @brief Calculate the theoretical maximum single memory allocation size per place for implicitly assembling the kernel matrix.
     * @param[in] num_features the total number of features
     * @param[in] num_classes the total number of classes
     * @return the theoretical maximum single memory allocation size per place for cg_implicit (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<memory_size> calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place(std::size_t num_features, std::size_t num_classes) const;
};

/**
 * @brief A rectangular data distribution where each place is responsible for approximately the same number of rows.
 */
class rectangular_data_distribution : public data_distribution {
  public:
    /**
     * @brief Calculate the data distribution (i.e., the number of rows in the kernel matrix a *place* is responsible for) such that each *place* has approximately the same number of data points it is responsible for.
     * @param[in] num_rows the number of data points to distribute
     * @param[in] num_places the number of places, i.e., different devices to distribute the data to
     */
    rectangular_data_distribution(std::size_t num_rows, std::size_t num_places);
};

}  // namespace plssvm::detail

template <>
struct fmt::formatter<plssvm::detail::triangular_data_distribution> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::rectangular_data_distribution> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_DATA_DISTRIBUTION_HPP_

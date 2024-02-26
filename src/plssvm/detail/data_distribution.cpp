/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/data_distribution.hpp"

#include "plssvm/constants.hpp"           // plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"       // PLSSVM_ASSERT
#include "plssvm/detail/memory_size.hpp"  // plssvm::detail::memory_size

#include <algorithm>  // std::max
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

[[nodiscard]] std::size_t calculate_data_set_num_entries(const std::size_t num_data_points, const std::size_t num_features) noexcept {
    return (num_data_points + plssvm::PADDING_SIZE) * (num_features + plssvm::PADDING_SIZE);
}

[[nodiscard]] std::size_t calculate_q_red_num_entries(const std::size_t num_data_points) noexcept {
    return num_data_points - 1 + plssvm::PADDING_SIZE;
}

[[nodiscard]] std::size_t calculate_blas_matrix_entries(const std::size_t num_data_points, const std::size_t num_classes) noexcept {
    return (num_data_points - 1 + plssvm::PADDING_SIZE) * (num_classes + plssvm::PADDING_SIZE);
}

namespace plssvm::detail {

using namespace literals;

std::vector<std::size_t> calculate_data_distribution_triangular(const std::size_t num_rows_reduced, const std::size_t num_places) {
    PLSSVM_ASSERT(num_rows_reduced > 0, "At least one row must be present!");
    PLSSVM_ASSERT(num_places > 0, "At least one place must be present!");

    std::vector<std::size_t> range(num_places + 1, num_rows_reduced);
    if (!range.empty()) {  // necessary to silence GCC "potential null pointer dereference [-Wnull-dereference]" warning
        range.front() = 0;
    }

    // only the upper triangular matrix is important
    const std::size_t balanced = (num_rows_reduced * (num_rows_reduced + 1) / 2) / num_places;

    std::size_t range_idx = 1;
    std::size_t sum = 0;
    std::size_t row = 0;

    // the first row has the most data points, while the last row has the fewest
    for (std::size_t i = num_rows_reduced; i >= 1; --i) {
        sum += i;
        ++row;
        if (sum >= balanced) {
            range[range_idx++] = row;
            sum = 0;
        }
    }

    return range;
}

std::vector<std::size_t> calculate_data_distribution_rectangular(const std::size_t num_rows, const std::size_t num_places) {
    PLSSVM_ASSERT(num_rows > 0, "At least one row must be present!");
    PLSSVM_ASSERT(num_places > 0, "At least one place must be present!");

    std::vector<std::size_t> res(num_places + 1, num_rows);
    const std::size_t balanced = num_rows / num_places;
    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        res[device_id] = balanced * device_id;
    }
    return res;
}

std::size_t get_place_specific_num_rows(const std::size_t place, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(place < data_distribution.size() - 1, "The queried place can at most be {}, but is {}!", data_distribution.size() - 1, place);
    return data_distribution[place + 1] - data_distribution[place];
}

std::size_t get_place_row_offset(const std::size_t place, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(place < data_distribution.size() - 1, "The queried place can at most be {}, but is {}!", data_distribution.size() - 1, place);
    return data_distribution[place];
}

std::size_t calculate_explicit_kernel_matrix_num_entries_padded(const std::size_t num_rows_reduced, const std::size_t place, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(num_rows_reduced > 0, "At least one row must be present!");
    PLSSVM_ASSERT(place < data_distribution.size() - 1, "The queried place can at most be {}, but is {}!", data_distribution.size() - 1, place);
    PLSSVM_ASSERT(data_distribution.size() >= 2, "Invalid data distribution!");

    /*
     *                    num_rows_reduced + PADDING
     *      ____________________________________________________
     *      |               |\                           .      |
     *      |               | \                          .      |
     *      |      A        |  \     num_entries_padded  .      |  device_specific_num_rows + PADDING
     *      |               | B \                        .      |
     *      |_______________|____\........................      |
     *          row_offset        \           PADDING           |
     *                             \____________________________|
     *
     *  A + B = values to discard
     */
    const std::size_t device_specific_num_rows = detail::get_place_specific_num_rows(place, data_distribution);
    const std::size_t row_offset = detail::get_place_row_offset(place, data_distribution);

    const std::size_t A = row_offset * (device_specific_num_rows + PADDING_SIZE);
    const std::size_t B = (device_specific_num_rows + PADDING_SIZE - 1) * (device_specific_num_rows + PADDING_SIZE) / 2;
    const std::size_t total = (device_specific_num_rows + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE);

    // calculate the number of matrix entries
    return total - A - B;
}

std::vector<memory_size> calculate_maximum_explicit_kernel_matrix_memory_needed_per_place(const std::size_t num_data_points, const std::size_t num_features, const std::size_t num_classes, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(num_data_points > 0, "At least one data point be present!");
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");
    PLSSVM_ASSERT(data_distribution.size() >= 2, "Invalid data distribution!");

    const std::size_t num_places = data_distribution.size() - 1;
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (get_place_specific_num_rows(device_id, data_distribution) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = calculate_data_set_num_entries(num_data_points, num_features);

        // the size of q_red
        const std::size_t q_red_size = calculate_q_red_num_entries(num_data_points);

        // the size of the explicitly stored kernel matrix
        const std::size_t kernel_matrix_size{ detail::calculate_explicit_kernel_matrix_num_entries_padded(num_data_points - 1, device_id, data_distribution) };

        // the B and C matrices for the explicit SYMM kernel
        std::size_t blas_matrices_size = 2 * calculate_blas_matrix_entries(num_data_points, num_classes);
        if (device_id == 0 && num_places > 1) {
            // device 0 has to save an additional matrix used to accumulate the partial results from the other devices
            blas_matrices_size += calculate_blas_matrix_entries(num_data_points, num_classes);
        }

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * (q_red_size + kernel_matrix_size + std::max(data_set_size, blas_matrices_size)) };
    }

    return res;
}

std::vector<memory_size> calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place(const std::size_t num_data_points, const std::size_t num_features, const std::size_t num_classes, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(num_data_points > 0, "At least one data point be present!");
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");
    PLSSVM_ASSERT(data_distribution.size() >= 2, "Invalid data distribution!");

    const std::size_t num_places = data_distribution.size() - 1;
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (get_place_specific_num_rows(device_id, data_distribution) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = calculate_data_set_num_entries(num_data_points, num_features);

        // the size of q_red including padding
        const std::size_t q_red_size = calculate_q_red_num_entries(num_data_points);

        // the size of the explicitly stored kernel matrix including padding
        const std::size_t kernel_matrix_size{ detail::calculate_explicit_kernel_matrix_num_entries_padded(num_data_points - 1, device_id, data_distribution) };

        // the size of the B or C matrix for the explicit SYMM kernel including padding
        const std::size_t blas_matrix_size = calculate_blas_matrix_entries(num_data_points, num_classes);

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * std::max({ data_set_size, q_red_size, kernel_matrix_size, blas_matrix_size }) };
    }

    return res;
}

std::vector<memory_size> calculate_maximum_implicit_kernel_matrix_memory_needed_per_place(const std::size_t num_data_points, const std::size_t num_features, const std::size_t num_classes, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(num_data_points > 0, "At least one data point be present!");
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");
    PLSSVM_ASSERT(data_distribution.size() >= 2, "Invalid data distribution!");

    const std::size_t num_places = data_distribution.size() - 1;
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (get_place_specific_num_rows(device_id, data_distribution) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = calculate_data_set_num_entries(num_data_points, num_features);

        // the size of q_red
        const std::size_t q_red_size = calculate_q_red_num_entries(num_data_points);

        // the B and C matrices for the explicit SYMM kernel
        std::size_t blas_matrices_size = 2 * calculate_blas_matrix_entries(num_data_points, num_classes);
        if (device_id == 0 && num_places > 1) {
            // device 0 has to save an additional matrix used to accumulate the partial results from the other devices
            blas_matrices_size += calculate_blas_matrix_entries(num_data_points, num_classes);
        }

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * (data_set_size + q_red_size + blas_matrices_size) };
    }

    return res;
}

std::vector<memory_size> calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place(const std::size_t num_data_points, const std::size_t num_features, const std::size_t num_classes, const std::vector<std::size_t> &data_distribution) {
    PLSSVM_ASSERT(num_data_points > 0, "At least one data point be present!");
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");
    PLSSVM_ASSERT(data_distribution.size() >= 2, "Invalid data distribution!");

    const std::size_t num_places = data_distribution.size() - 1;
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (get_place_specific_num_rows(device_id, data_distribution) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = calculate_data_set_num_entries(num_data_points, num_features);

        // the size of q_red including padding
        const std::size_t q_red_size = calculate_q_red_num_entries(num_data_points);

        // the size of the B or C matrix for the explicit SYMM kernel including padding
        const std::size_t blas_matrix_size = calculate_blas_matrix_entries(num_data_points, num_classes);

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * std::max({ data_set_size, q_red_size, blas_matrix_size }) };
    }

    return res;
}

}  // namespace plssvm::detail

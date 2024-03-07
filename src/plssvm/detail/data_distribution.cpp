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

#include "fmt/format.h"  // fmt::format, fmt::join

#include <algorithm>  // std::max, std::fill
#include <cstddef>    // std::size_t
#include <ostream>    // std::ostream
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

data_distribution::data_distribution(const std::size_t num_rows, const std::size_t num_places) :
    distribution_(num_places + 1),
    num_rows_{ num_rows },
    num_places_{ num_places } {
    PLSSVM_ASSERT(num_rows_ > 0, "At least one row must be present!");
    PLSSVM_ASSERT(num_places_ > 0, "At least one place must be present!");
}

data_distribution::~data_distribution() = default;

std::size_t data_distribution::place_specific_num_rows(const std::size_t place) const noexcept {
    PLSSVM_ASSERT(distribution_.size() >= 2, "At least one place must be present and, therefore, the distribution vector must contain at least two entries!");
    PLSSVM_ASSERT(place < distribution_.size() - 1, "The queried place can at most be {}, but is {}!", distribution_.size() - 1, place);
    return distribution_[place + 1] - distribution_[place];
}

std::size_t data_distribution::place_row_offset(const std::size_t place) const noexcept {
    PLSSVM_ASSERT(distribution_.size() >= 2, "At least one place must be present and, therefore, the distribution vector must contain at least two entries!");
    PLSSVM_ASSERT(place < distribution_.size() - 1, "The queried place can at most be {}, but is {}!", distribution_.size() - 1, place);
    return distribution_[place];
}

const std::vector<std::size_t> &data_distribution::distribution() const noexcept {
    return distribution_;
}

std::size_t data_distribution::num_rows() const noexcept {
    return num_rows_;
}

std::size_t data_distribution::num_places() const noexcept {
    return num_places_;
}

std::ostream &operator<<(std::ostream &out, const data_distribution &dist) {
    return out << fmt::format("{ num_rows: {}, num_places: {}, dist: [{}] }", dist.num_rows(), dist.num_places(), fmt::join(dist.distribution(), ", "));
}

//*************************************************************************************************************************************//
//                                                     specific data distributions                                                     //
//*************************************************************************************************************************************//
using namespace literals;

triangular_data_distribution::triangular_data_distribution(const std::size_t num_rows, const std::size_t num_places) :
    data_distribution{ num_rows, num_places } {
    // set all distribution values to "num_rows"
    std::fill(distribution_.begin(), distribution_.end(), num_rows);

    if (!distribution_.empty()) {  // necessary to silence GCC "potential null pointer dereference [-Wnull-dereference]" warning
        distribution_.front() = 0;
    }

    // only the upper triangular matrix is important
    const std::size_t balanced = (num_rows * (num_rows + 1) / 2) / num_places;

    std::size_t range_idx = 1;
    std::size_t sum = 0;
    std::size_t row = 0;

    // the first row has the most data points, while the last row has the fewest
    for (std::size_t i = num_rows; i >= 1; --i) {
        sum += i;
        ++row;
        if (sum >= balanced) {
            distribution_[range_idx++] = row;
            sum = 0;
        }
    }

    PLSSVM_ASSERT(std::is_sorted(distribution_.cbegin(), distribution_.cend()), "The distribution must be sorted in an ascending order!");
}

std::size_t triangular_data_distribution::calculate_explicit_kernel_matrix_num_entries_padded(const std::size_t place) const noexcept {
    PLSSVM_ASSERT(place < distribution_.size() - 1, "The queried place can at most be {}, but is {}!", distribution_.size() - 1, place);

    /*
     *                        num_rows_ + PADDING
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
    const std::size_t device_specific_num_rows = this->place_specific_num_rows(place);
    const std::size_t row_offset = this->place_row_offset(place);

    const std::size_t A = row_offset * (device_specific_num_rows + PADDING_SIZE);
    const std::size_t B = (device_specific_num_rows + PADDING_SIZE - 1) * (device_specific_num_rows + PADDING_SIZE) / 2;
    const std::size_t total = (device_specific_num_rows + PADDING_SIZE) * (this->num_rows() + PADDING_SIZE);

    // calculate the number of matrix entries
    return total - A - B;
}

std::vector<memory_size> triangular_data_distribution::calculate_maximum_explicit_kernel_matrix_memory_needed_per_place(const std::size_t num_features, const std::size_t num_classes) const {
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");

    const std::size_t num_places = this->num_places();
    const std::size_t num_rows = this->num_rows() + 1;  // account for dimensional reduction
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (this->place_specific_num_rows(device_id) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = ::calculate_data_set_num_entries(num_rows, num_features);

        // the size of q_red
        const std::size_t q_red_size = ::calculate_q_red_num_entries(num_rows);

        // the size of the explicitly stored kernel matrix
        const std::size_t kernel_matrix_size{ this->calculate_explicit_kernel_matrix_num_entries_padded(device_id) };

        // the B and C matrices for the explicit SYMM kernel
        std::size_t blas_matrices_size = 2 * ::calculate_blas_matrix_entries(num_rows, num_classes);
        if (device_id == 0 && num_places > 1) {
            // device 0 has to save an additional matrix used to accumulate the partial results from the other devices
            blas_matrices_size += ::calculate_blas_matrix_entries(num_rows, num_classes);
        }

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * (q_red_size + kernel_matrix_size + std::max(data_set_size, blas_matrices_size)) };
    }

    return res;
}

std::vector<memory_size> triangular_data_distribution::calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place(const std::size_t num_features, const std::size_t num_classes) const {
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");

    const std::size_t num_places = this->num_places();
    const std::size_t num_rows = this->num_rows() + 1;  // account for dimensional reduction
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (this->place_specific_num_rows(device_id) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = ::calculate_data_set_num_entries(num_rows, num_features);

        // the size of q_red including padding
        const std::size_t q_red_size = ::calculate_q_red_num_entries(num_rows);

        // the size of the explicitly stored kernel matrix including padding
        const std::size_t kernel_matrix_size{ this->calculate_explicit_kernel_matrix_num_entries_padded(device_id) };

        // the size of the B or C matrix for the explicit SYMM kernel including padding
        const std::size_t blas_matrix_size = ::calculate_blas_matrix_entries(num_rows, num_classes);

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * std::max({ data_set_size, q_red_size, kernel_matrix_size, blas_matrix_size }) };
    }

    return res;
}

std::vector<memory_size> triangular_data_distribution::calculate_maximum_implicit_kernel_matrix_memory_needed_per_place(const std::size_t num_features, const std::size_t num_classes) const {
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");

    const std::size_t num_places = this->num_places();
    const std::size_t num_rows = this->num_rows() + 1;  // account for dimensional reduction
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (this->place_specific_num_rows(device_id) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = ::calculate_data_set_num_entries(num_rows, num_features);

        // the size of q_red
        const std::size_t q_red_size = ::calculate_q_red_num_entries(num_rows);

        // the B and C matrices for the explicit SYMM kernel
        std::size_t blas_matrices_size = 2 * ::calculate_blas_matrix_entries(num_rows, num_classes);
        if (device_id == 0 && num_places > 1) {
            // device 0 has to save an additional matrix used to accumulate the partial results from the other devices
            blas_matrices_size += ::calculate_blas_matrix_entries(num_rows, num_classes);
        }

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * (data_set_size + q_red_size + blas_matrices_size) };
    }

    return res;
}

std::vector<memory_size> triangular_data_distribution::calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place(const std::size_t num_features, const std::size_t num_classes) const {
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be present!");
    PLSSVM_ASSERT(num_classes > 0, "At least two classes must be present!");

    const std::size_t num_places = this->num_places();
    const std::size_t num_rows = this->num_rows() + 1;  // account for dimensional reduction
    std::vector<memory_size> res(num_places, 0_B);

    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (this->place_specific_num_rows(device_id) == 0) {
            continue;
        }

        // data set including padding
        const std::size_t data_set_size = ::calculate_data_set_num_entries(num_rows, num_features);

        // the size of q_red including padding
        const std::size_t q_red_size = ::calculate_q_red_num_entries(num_rows);

        // the size of the B or C matrix for the explicit SYMM kernel including padding
        const std::size_t blas_matrix_size = ::calculate_blas_matrix_entries(num_rows, num_classes);

        // add up the individual sizes and report the memory size in BYTES
        res[device_id] = memory_size{ sizeof(real_type) * std::max({ data_set_size, q_red_size, blas_matrix_size }) };
    }

    return res;
}

rectangular_data_distribution::rectangular_data_distribution(const std::size_t num_rows, const std::size_t num_places) :
    data_distribution{ num_rows, num_places } {
    // uniform distribution
    const std::size_t balanced = num_rows / num_places;
    for (std::size_t device_id = 0; device_id < num_places; ++device_id) {
        distribution_[device_id] = balanced * device_id;
    }

    // fill remaining values into distribution starting at device 0
    const std::size_t remaining = num_rows - num_places * balanced;
    std::size_t running = 0;
    for (std::size_t device_id = 1; device_id <= num_places; ++device_id) {
        distribution_[device_id] += running;
        if (device_id - 1 < remaining) {
            distribution_[device_id] += 1;
            ++running;
        }
    }
    distribution_.back() = num_rows;

    PLSSVM_ASSERT(std::is_sorted(distribution_.cbegin(), distribution_.cend()), "The distribution must be sorted in an ascending order!");
}

}  // namespace plssvm::detail

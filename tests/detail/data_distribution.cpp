/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the multi-GPU data distribution implementation.
 */

#include "plssvm/detail/data_distribution.hpp"

#include "plssvm/constants.hpp"           // plssvm::PADDING_SIZE
#include "plssvm/detail/memory_size.hpp"  // plssvm::detail::memory_size

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <algorithm>  // std::is_sorted
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

using namespace plssvm::detail::literals;

//*************************************************************************************************************************************//
//                                                    triangular data distributions                                                    //
//*************************************************************************************************************************************//

TEST(TriangularDataDistribution, construct) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // test getter
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 5);
    EXPECT_EQ(dist_vec.front(), 0);                                   // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 1024);                                 // the distribution must end with the number of rows
    EXPECT_TRUE(std::is_sorted(dist_vec.cbegin(), dist_vec.cend()));  // the distribution values must be sorted in ascending order
    EXPECT_EQ(dist.num_rows(), 1024);
    EXPECT_EQ(dist.num_places(), 4);
}

TEST(TriangularDataDistribution, place_specific_num_rows) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the place specific number of rows calculation for sanity
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_LT(dist.place_specific_num_rows(place), 1024);
    }
}

TEST(TriangularDataDistribution, place_row_offset) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the place specific row offset calculation
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GE(dist.place_specific_num_rows(place), 0);
        EXPECT_LT(dist.place_specific_num_rows(place), 1024);
    }
}

TEST(TriangularDataDistribution, distribution) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the distribution for sanity
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 5);
    EXPECT_EQ(dist_vec.front(), 0);                                   // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 1024);                                 // the distribution must end with the number of rows
    EXPECT_TRUE(std::is_sorted(dist_vec.cbegin(), dist_vec.cend()));  // the distribution values must be sorted in ascending order
}

TEST(TriangularDataDistribution, distribution_one_place) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 1 };

    // check the distribution for sanity
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 2);
    EXPECT_EQ(dist_vec.front(), 0);    // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 1024);  // the distribution must end with the number of rows
}

TEST(TriangularDataDistribution, distribution_fewer_rows_than_places) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 6, 8 };

    // check the distribution for sanity
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 9);
    EXPECT_EQ(dist_vec.front(), 0);                                   // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 6);                                    // the distribution must end with the number of rows
    EXPECT_TRUE(std::is_sorted(dist_vec.cbegin(), dist_vec.cend()));  // the distribution values must be sorted in ascending order
}

TEST(TriangularDataDistribution, num_rows) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the number of rows getter
    EXPECT_EQ(dist.num_rows(), 1024);
}

TEST(TriangularDataDistribution, num_places) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the number of places getter
    EXPECT_EQ(dist.num_places(), 4);
}

TEST(TriangularDataDistribution, calculate_explicit_kernel_matrix_num_entries_padded) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the returned values
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GE(dist.calculate_explicit_kernel_matrix_num_entries_padded(place), 0);
        EXPECT_LT(dist.calculate_explicit_kernel_matrix_num_entries_padded(place), (1024 + plssvm::PADDING_SIZE) * (1024 + plssvm::PADDING_SIZE));  // must be less than the squared matrix padded
    }
}

TEST(TriangularDataDistribution, calculate_maximum_explicit_kernel_matrix_memory_needed_per_place) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the returned values
    const std::vector<plssvm::detail::memory_size> ret = dist.calculate_maximum_explicit_kernel_matrix_memory_needed_per_place(128, 32);
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GT(ret[place], 0_B);
    }
}

TEST(TriangularDataDistribution, calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the returned values
    const std::vector<plssvm::detail::memory_size> ret = dist.calculate_maximum_explicit_kernel_matrix_memory_allocation_size_per_place(128, 32);
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GT(ret[place], 0_B);
    }
}

TEST(TriangularDataDistribution, calculate_maximum_implicit_kernel_matrix_memory_needed_per_place) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the returned values
    const std::vector<plssvm::detail::memory_size> ret = dist.calculate_maximum_implicit_kernel_matrix_memory_needed_per_place(128, 32);
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GT(ret[place], 0_B);
    }
}

TEST(TriangularDataDistribution, calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place) {
    // create a triangular data distribution
    const plssvm::detail::triangular_data_distribution dist{ 1024, 4 };

    // check the returned values
    const std::vector<plssvm::detail::memory_size> ret = dist.calculate_maximum_implicit_kernel_matrix_memory_allocation_size_per_place(128, 32);
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GT(ret[place], 0_B);
    }
}

//*************************************************************************************************************************************//
//                                                    rectangular data distributions                                                   //
//*************************************************************************************************************************************//

TEST(RectangularDataDistribution, construct) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 4 };

    // test getter
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 5);
    EXPECT_EQ(dist_vec.front(), 0);                                   // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 1024);                                 // the distribution must end with the number of rows
    EXPECT_TRUE(std::is_sorted(dist_vec.cbegin(), dist_vec.cend()));  // the distribution values must be sorted in ascending order
    EXPECT_EQ(dist.num_rows(), 1024);
    EXPECT_EQ(dist.num_places(), 4);
}

TEST(RectangularDataDistribution, place_specific_num_rows) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 4 };

    // check the place specific number of rows calculation for sanity
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_LT(dist.place_specific_num_rows(place), 1024);
    }
}

TEST(RectangularDataDistribution, place_row_offset) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 4 };

    // check the place specific row offset calculation
    for (std::size_t place = 0; place < dist.num_places(); ++place) {
        EXPECT_GE(dist.place_specific_num_rows(place), 0);
        EXPECT_LT(dist.place_specific_num_rows(place), 1024);
    }
}

TEST(RectangularDataDistribution, distribution) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 4 };

    // check the distribution for sanity
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 5);
    EXPECT_EQ(dist_vec.front(), 0);                                   // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 1024);                                 // the distribution must end with the number of rows
    EXPECT_TRUE(std::is_sorted(dist_vec.cbegin(), dist_vec.cend()));  // the distribution values must be sorted in ascending order
}

TEST(RectangularDataDistribution, distribution_one_place) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 1 };

    // check the distribution for sanity
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 2);
    EXPECT_EQ(dist_vec.front(), 0);    // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 1024);  // the distribution must end with the number of rows
}

TEST(RectangularDataDistribution, distribution_fewer_rows_than_places) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 6, 8 };

    // check the distribution for sanity
    const std::vector<std::size_t> dist_vec = dist.distribution();
    EXPECT_EQ(dist_vec.size(), 9);
    EXPECT_EQ(dist_vec.front(), 0);                                   // the distribution must start with 0
    EXPECT_EQ(dist_vec.back(), 6);                                    // the distribution must end with the number of rows
    EXPECT_TRUE(std::is_sorted(dist_vec.cbegin(), dist_vec.cend()));  // the distribution values must be sorted in ascending order
}

TEST(RectangularDataDistribution, num_rows) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 4 };

    // check the number of rows getter
    EXPECT_EQ(dist.num_rows(), 1024);
}

TEST(RectangularDataDistribution, num_places) {
    // create a triangular data distribution
    const plssvm::detail::rectangular_data_distribution dist{ 1024, 4 };

    // check the number of places getter
    EXPECT_EQ(dist.num_places(), 4);
}

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for MPI version functions.
 * @note Assumes only a single MPI rank, since more are **not** supported in our tests!
 */

#include "plssvm/mpi/detail/version.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_FALSE

#include <string>  // std::string

TEST(MPIVersion, MPILibraryVersionNotEmpty) {
    // the MPI library version may not be empty
    EXPECT_FALSE(plssvm::mpi::detail::mpi_library_version().empty());
}

TEST(MPIVersion, MPIVersionNotEmpty) {
    // the MPI version may not be empty
    EXPECT_FALSE(plssvm::mpi::detail::mpi_version().empty());
}

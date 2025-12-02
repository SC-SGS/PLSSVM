/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for MPI utility functions.
 * @note Assumes only a single MPI rank, since more are **not** supported in our tests!
 */

#include "plssvm/mpi/detail/utility.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_SUCCESS, MPI_ERR_COMM
#endif

#include "gtest/gtest.h"  // TEST, EXPECT_FALSE, EXPECT_THROW, EXPECT_NO_THROW

#include <string>  // std::string

TEST(MPIUtility, MPIErrorCheck) {
    // test error check macro
#if defined(PLSSVM_HAS_MPI_ENABLED)
    // if MPI is enabled, MPI_SUCCESS may never throw
    EXPECT_NO_THROW(plssvm::mpi::detail::mpi_error_check(MPI_SUCCESS));

    // if MPI is enabled, MPI_ERR_COMM must throw
    EXPECT_THROW(plssvm::mpi::detail::mpi_error_check(MPI_ERR_COMM), plssvm::mpi_exception);
#else
    // if MPI is disabled, may never throw
    EXPECT_NO_THROW(plssvm::mpi::detail::mpi_error_check(1));
#endif
}

TEST(MPIUtility, NodeName) {
    // the MPI node name may not be empty
    EXPECT_FALSE(plssvm::mpi::detail::node_name().empty());
}

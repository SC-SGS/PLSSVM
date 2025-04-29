/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for MPI environment wrapper functions.
 * @note Assumes only a single MPI rank, since more are **not** supported in our tests!
 */

#include "plssvm/mpi/environment.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_FALSE, EXPECT_DEATH

TEST(MPIEnvironment, is_executed_via_mpirun) {
    // since we do not support mpirun ctest, the function must return false
    EXPECT_FALSE(plssvm::mpi::is_executed_via_mpirun());
}

TEST(MPIEnvironmentDeathTest, abort_world) {
    // test whether the abort function fires correctly
    EXPECT_DEATH(plssvm::mpi::abort_world(), "");
}

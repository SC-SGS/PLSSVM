/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the Kokkos `constexpr_available_execution_spaces()` function.
 */

#include "plssvm/backends/Kokkos/detail/constexpr_available_execution_spaces.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_FALSE

TEST(KokkosConstexprAvailableExecutionSpaces, constexpr_available_execution_spaces) {
    // at least one execution space must always be available
    EXPECT_FALSE(plssvm::kokkos::detail::constexpr_available_execution_spaces().empty());
}

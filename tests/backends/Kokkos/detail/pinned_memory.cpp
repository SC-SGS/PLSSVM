/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the Kokkos backend pinned memory.
 */

#include "plssvm/backends/Kokkos/detail/pinned_memory.hpp"  // plssvm::kokkos::detail::pinned_memory

#include "tests/backends/generic_pinned_memory_tests.hpp"  // generic pinned memory tests to instantiate
#include "tests/naming.hpp"                                // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                         // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T>
struct kokkos_pinned_memory_test_type {
    using pinned_memory_type = plssvm::kokkos::detail::pinned_memory<T>;

    constexpr static bool can_pin = false;  // TODO: try implementing in Kokkos?
};

using kokkos_pinned_memory_tuple = std::tuple<kokkos_pinned_memory_test_type<float>, kokkos_pinned_memory_test_type<double>>;

// the tests used in the instantiated GTest test suites
using kokkos_pinned_memory_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<kokkos_pinned_memory_tuple>>;
using kokkos_pinned_memory_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<kokkos_pinned_memory_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosPinnedMemory, PinnedMemory, kokkos_pinned_memory_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosPinnedMemory, PinnedMemoryLayout, kokkos_pinned_memory_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(KokkosPinnedMemoryDeathTest, PinnedMemoryDeathTest, kokkos_pinned_memory_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosPinnedMemoryDeathTest, PinnedMemoryLayoutDeathTest, kokkos_pinned_memory_layout_type_gtest, naming::test_parameter_to_name);

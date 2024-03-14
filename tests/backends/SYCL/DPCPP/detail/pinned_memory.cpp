/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the CUDA backend pinned memory.
 */

#include "plssvm/backends/SYCL/DPCPP/detail/pinned_memory.hpp"  // plssvm::dpcpp::detail::pinned_memory

#include "tests/backends/generic_pinned_memory_tests.hpp"  // generic pinned memory tests to instantiate
#include "tests/naming.hpp"                                // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                         // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T>
struct dpcpp_pinned_memory_test_type {
    using pinned_memory_type = plssvm::dpcpp::detail::pinned_memory<T>;

    constexpr static bool can_pin = false;
};

using dpcpp_pinned_memory_tuple = std::tuple<dpcpp_pinned_memory_test_type<float>, dpcpp_pinned_memory_test_type<double>>;

// the tests used in the instantiated GTest test suites
using dpcpp_pinned_memory_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_pinned_memory_tuple>>;
using dpcpp_pinned_memory_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_pinned_memory_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPPinnedMemory, PinnedMemory, dpcpp_pinned_memory_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPPinnedMemory, PinnedMemoryLayout, dpcpp_pinned_memory_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPPinnedMemoryDeathTest, PinnedMemoryDeathTest, dpcpp_pinned_memory_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPPinnedMemoryDeathTest, PinnedMemoryLayoutDeathTest, dpcpp_pinned_memory_layout_type_gtest, naming::test_parameter_to_name);

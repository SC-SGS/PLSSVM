/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the CUDA backend pinned memory.
 */

#include "plssvm/backends/CUDA/detail/pinned_memory.cuh"  // plssvm::cuda::detail::pinned_memory

#include "tests/backends/generic_pinned_memory_tests.hpp"  // generic pinned memory tests to instantiate
#include "tests/naming.hpp"                                // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                         // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T>
struct cuda_pinned_memory_test_type {
    using pinned_memory_type = plssvm::cuda::detail::pinned_memory<T>;

    constexpr static bool can_pin = true;
};

using cuda_pinned_memory_tuple = std::tuple<cuda_pinned_memory_test_type<float>, cuda_pinned_memory_test_type<double>>;

// the tests used in the instantiated GTest test suites
using cuda_pinned_memory_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<cuda_pinned_memory_tuple>>;
using cuda_pinned_memory_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<cuda_pinned_memory_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(CUDAPinnedMemory, PinnedMemory, cuda_pinned_memory_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDAPinnedMemory, PinnedMemoryLayout, cuda_pinned_memory_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(CUDAPinnedMemoryDeathTest, PinnedMemoryDeathTest, cuda_pinned_memory_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDAPinnedMemoryDeathTest, PinnedMemoryLayoutDeathTest, cuda_pinned_memory_layout_type_gtest, naming::test_parameter_to_name);

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic device_ptr tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_PINNED_MEMORY_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_PINNED_MEMORY_TESTS_HPP_
#pragma once

#include "plssvm/matrix.hpp"  // plssvm::matrix, plssvm::layout_type

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_MATRIX_EQ
#include "tests/types_to_test.hpp"       // util::test_parameter_type_at_t
#include "tests/utility.hpp"

#include "gtest//gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_EQ, EXPECT_DEATH

#include <tuple>   // std::ignore
#include <vector>  // std::vector

template <typename T>
class PinnedMemory : public ::testing::Test {
  protected:
    using fixture_test_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE_P(PinnedMemory);

TYPED_TEST_P(PinnedMemory, construct_vector) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr bool can_pin = test_type::can_pin;

    const std::vector<real_type> vec(42);

    // try pinning memory
    const pinned_memory_type mem{ vec };

    // check if the memory could successfully be pinned
    EXPECT_EQ(mem.is_pinned(), can_pin);
}

TYPED_TEST_P(PinnedMemory, construct_pointer_and_size) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr bool can_pin = test_type::can_pin;

    const std::vector<real_type> vec(42);

    // try pinning memory
    const pinned_memory_type mem{ vec.data(), vec.size() };

    // check if the memory could successfully be pinned
    EXPECT_EQ(mem.is_pinned(), can_pin);
}

TYPED_TEST_P(PinnedMemory, is_pinned) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr bool can_pin = test_type::can_pin;

    const std::vector<real_type> vec(42);

    // try pinning memory
    const pinned_memory_type mem{ vec };

    // check if the is_pinned getter works
    EXPECT_EQ(mem.is_pinned(), can_pin);
}

REGISTER_TYPED_TEST_SUITE_P(PinnedMemory, construct_vector, construct_pointer_and_size, is_pinned);

template <typename T>
class PinnedMemoryLayout : public PinnedMemory<T> {
  protected:
    using typename PinnedMemory<T>::fixture_test_type;
};

TYPED_TEST_SUITE_P(PinnedMemoryLayout);

TYPED_TEST_P(PinnedMemoryLayout, construct_matrix) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool can_pin = test_type::can_pin;

    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try pinning memory
    const pinned_memory_type mem{ matr };

    // check if the memory could successfully be pinned
    EXPECT_EQ(mem.is_pinned(), can_pin);
}

REGISTER_TYPED_TEST_SUITE_P(PinnedMemoryLayout, construct_matrix);

template <typename T>
class PinnedMemoryDeathTest : public PinnedMemory<T> {
  protected:
    using fixture_test_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE_P(PinnedMemoryDeathTest);

TYPED_TEST_P(PinnedMemoryDeathTest, construct_empty_vector) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr bool can_pin = test_type::can_pin;

    const std::vector<real_type> vec{};

    // try pinning empty memory doesn't work -> only if pinning is possible!
    if constexpr (can_pin) {
        EXPECT_DEATH(std::ignore = pinned_memory_type{ vec }, "Can't pin a 0 B memory!");
    }
}

TYPED_TEST_P(PinnedMemoryDeathTest, construct_empty_pointer_and_size) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr bool can_pin = test_type::can_pin;

    const std::vector<real_type> vec{};

    // try pinning empty memory doesn't work -> only if pinning is possible!
    if constexpr (can_pin) {
        EXPECT_DEATH((std::ignore = pinned_memory_type{ vec.data(), vec.size() }), "Can't pin a 0 B memory!");
    }
}

TYPED_TEST_P(PinnedMemoryDeathTest, construct_nullptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    constexpr bool can_pin = test_type::can_pin;

    // try pinning empty memory doesn't work -> only if pinning is possible!
    if constexpr (can_pin) {
        EXPECT_DEATH((std::ignore = pinned_memory_type{ nullptr, 1 }), "ptr_ may not be the nullptr!");
    }
}

REGISTER_TYPED_TEST_SUITE_P(PinnedMemoryDeathTest, construct_empty_vector, construct_empty_pointer_and_size, construct_nullptr);

template <typename T>
class PinnedMemoryLayoutDeathTest : public PinnedMemory<T> {
  protected:
    using fixture_test_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE_P(PinnedMemoryLayoutDeathTest);

TYPED_TEST_P(PinnedMemoryLayoutDeathTest, construct_empty_matrix) {
    using test_type = typename TestFixture::fixture_test_type;
    using pinned_memory_type = typename test_type::pinned_memory_type;
    using real_type = typename pinned_memory_type::value_type;
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool can_pin = test_type::can_pin;

    const plssvm::matrix<real_type, layout> matr{};

    // try pinning empty memory doesn't work -> only if pinning is possible!
    if constexpr (can_pin) {
        EXPECT_DEATH(std::ignore = pinned_memory_type{ matr }, "Can't pin a 0 B memory!");
    }
}

REGISTER_TYPED_TEST_SUITE_P(PinnedMemoryLayoutDeathTest, construct_empty_matrix);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_PINNED_MEMORY_TESTS_HPP_

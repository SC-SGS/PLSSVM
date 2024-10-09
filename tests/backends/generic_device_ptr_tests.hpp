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

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_DEVICE_PTR_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_DEVICE_PTR_TESTS_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::gpu_device_ptr_exception
#include "plssvm/matrix.hpp"                 // plssvm::matrix
#include "plssvm/shape.hpp"                  // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_MATRIX_EQ
#include "tests/types_to_test.hpp"       // util::test_parameter_type_at_t
#include "tests/utility.hpp"

#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EQ, EXPECT_NE, EXPECT_DEATH
                          // ::testing::Test

#include <cstring>  // std::memset
#include <utility>  // std::move, std::swap
#include <vector>   // std::vector

template <typename T>
class DevicePtr : public ::testing::Test {
  protected:
    using fixture_test_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE_P(DevicePtr);

TYPED_TEST_P(DevicePtr, default_construct) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;

    // default construct device_ptr
    const device_ptr_type ptr{};

    // empty data
    EXPECT_FALSE(static_cast<bool>(ptr));
    EXPECT_EQ(ptr.get(), data_ptr_type{});
    EXPECT_EQ(ptr.size(), 0);
    EXPECT_EQ(ptr.shape(), (plssvm::shape{ 0, 0 }));
    EXPECT_TRUE(ptr.empty());
}

TYPED_TEST_P(DevicePtr, construct_size) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr{ 42, queue, use_usm_allocations };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_NE(ptr.get(), data_ptr_type{});
    EXPECT_EQ(ptr.shape(), (plssvm::shape{ 42, 1 }));
    // check padding
    EXPECT_EQ(ptr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(ptr.shape_padded(), (plssvm::shape{ 42, 1 }));
}

TYPED_TEST_P(DevicePtr, construct_shape) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr{ plssvm::shape{ 42, 16 }, queue, use_usm_allocations };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_NE(ptr.get(), data_ptr_type{});
    EXPECT_EQ(ptr.shape(), (plssvm::shape{ 42, 16 }));
    // check padding
    EXPECT_EQ(ptr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(ptr.shape_padded(), (plssvm::shape{ 42, 16 }));
}

TYPED_TEST_P(DevicePtr, construct_shape_and_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr{ plssvm::shape{ 42, 16 }, plssvm::shape{ 4, 4 }, queue, use_usm_allocations };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_NE(ptr.get(), data_ptr_type{});
    EXPECT_EQ(ptr.shape(), (plssvm::shape{ 42, 16 }));
    // check padding
    EXPECT_EQ(ptr.padding(), (plssvm::shape{ 4, 4 }));
    EXPECT_EQ(ptr.shape_padded(), (plssvm::shape{ 46, 20 }));
}

TYPED_TEST_P(DevicePtr, move_construct) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type first{ 42, queue, use_usm_allocations };
    const device_ptr_type second{ std::move(first) };

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    // EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 1 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 42, 1 }));

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, move_construct_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type first{ plssvm::shape{ 42, 10 }, plssvm::shape{ 4, 5 }, queue, use_usm_allocations };
    const device_ptr_type second{ std::move(first) };

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    // EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 10 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 46, 15 }));

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, move_assign) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type first{ 42, queue, use_usm_allocations };
    device_ptr_type second;

    // move assign
    second = std::move(first);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 1 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 42, 1 }));

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, move_assign_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type first{ plssvm::shape{ 42, 10 }, plssvm::shape{ 4, 5 }, queue, use_usm_allocations };
    device_ptr_type second;

    // move assign
    second = std::move(first);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 10 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 46, 15 }));

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, swap_member_function) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct two device_ptr
    device_ptr_type first{ 42, queue, use_usm_allocations };
    device_ptr_type second{};

    // swap both device_ptr using the member function
    first.swap(second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 1 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 42, 1 }));

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, swap_member_function_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct two device_ptr
    device_ptr_type first{ plssvm::shape{ 42, 10 }, plssvm::shape{ 4, 5 }, queue, use_usm_allocations };
    device_ptr_type second{};

    // swap both device_ptr using the member function
    first.swap(second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 10 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 46, 15 }));

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, swap_free_function) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct two device_ptr
    device_ptr_type first{ 42, queue, use_usm_allocations };
    device_ptr_type second{};

    // swap both device_ptr using the free function
    using std::swap;
    swap(first, second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 1 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 42, 1 }));

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, swap_free_function_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using data_ptr_type = typename device_ptr_type::device_pointer_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct two device_ptr
    device_ptr_type first{ plssvm::shape{ 42, 10 }, plssvm::shape{ 4, 5 }, queue, use_usm_allocations };
    device_ptr_type second{};

    // swap both device_ptr using the free function
    using std::swap;
    swap(first, second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), data_ptr_type{});
    EXPECT_EQ(second.shape(), (plssvm::shape{ 42, 10 }));
    // check padding
    EXPECT_EQ(second.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(second.shape_padded(), (plssvm::shape{ 46, 15 }));

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), data_ptr_type{});
    EXPECT_EQ(first.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(first.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(first.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, operator_bool) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue, use_usm_allocations };
    EXPECT_TRUE(static_cast<bool>(ptr1));

    // construct empty device_ptr
    const device_ptr_type ptr2{};
    EXPECT_FALSE(static_cast<bool>(ptr2));
}

TYPED_TEST_P(DevicePtr, size) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue, use_usm_allocations };
    EXPECT_EQ(ptr1.size(), 42);

    // construct device_ptr with shape
    const device_ptr_type ptr2{ plssvm::shape{ 42, 16 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr2.size(), 42 * 16);

    // construct device_ptr with shape and padding
    const device_ptr_type ptr3{ plssvm::shape{ 42, 16 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr3.size(), 42 * 16);

    // construct empty device_ptr
    const device_ptr_type ptr4{};
    EXPECT_EQ(ptr4.size(), 0);
}

TYPED_TEST_P(DevicePtr, shape) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue, use_usm_allocations };
    EXPECT_EQ(ptr1.shape(), (plssvm::shape{ 42, 1 }));

    // construct device_ptr with shape
    const device_ptr_type ptr2{ plssvm::shape{ 42, 16 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr2.shape(), (plssvm::shape{ 42, 16 }));

    // construct device_ptr with shape and padding
    const device_ptr_type ptr3{ plssvm::shape{ 42, 16 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr3.shape(), (plssvm::shape{ 42, 16 }));

    // construct empty device_ptr
    const device_ptr_type ptr4{};
    EXPECT_EQ(ptr4.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, empty) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue, use_usm_allocations };
    EXPECT_FALSE(ptr1.empty());

    // construct device_ptr
    const device_ptr_type ptr2{ plssvm::shape{ 42, 16 }, queue, use_usm_allocations };
    EXPECT_FALSE(ptr2.empty());

    // construct device_ptr
    const device_ptr_type ptr3{ plssvm::shape{ 42, 16 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_FALSE(ptr3.empty());

    // construct device_ptr
    const device_ptr_type ptr4{ plssvm::shape{ 0, 0 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_TRUE(ptr4.empty());

    // construct empty device_ptr
    const device_ptr_type ptr5{};
    EXPECT_TRUE(ptr5.empty());
}

TYPED_TEST_P(DevicePtr, padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr{ plssvm::shape{ 42, 16 }, plssvm::shape{ 4, 5 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr.padding(), (plssvm::shape{ 4, 5 }));
    ;
}

TYPED_TEST_P(DevicePtr, size_padded) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr{ plssvm::shape{ 42, 16 }, plssvm::shape{ 4, 5 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr.size_padded(), (42 + 4) * (16 + 5));
}

TYPED_TEST_P(DevicePtr, shape_padded) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue, use_usm_allocations };
    EXPECT_EQ(ptr1.shape_padded(), (plssvm::shape{ 42, 1 }));

    // construct device_ptr with shape
    const device_ptr_type ptr2{ plssvm::shape{ 42, 16 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr2.shape_padded(), (plssvm::shape{ 42, 16 }));

    // construct device_ptr with shape and padding
    const device_ptr_type ptr3{ plssvm::shape{ 42, 16 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_EQ(ptr3.shape_padded(), (plssvm::shape{ 45, 19 }));

    // construct empty device_ptr
    const device_ptr_type ptr4{};
    EXPECT_EQ(ptr4.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST_P(DevicePtr, is_padded) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue, use_usm_allocations };
    EXPECT_FALSE(ptr1.is_padded());

    // construct device_ptr
    const device_ptr_type ptr2{ plssvm::shape{ 42, 16 }, queue, use_usm_allocations };
    EXPECT_FALSE(ptr2.is_padded());

    // construct device_ptr
    const device_ptr_type ptr3{ plssvm::shape{ 42, 16 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_TRUE(ptr3.is_padded());

    // construct device_ptr
    const device_ptr_type ptr4{ plssvm::shape{ 0, 0 }, plssvm::shape{ 3, 3 }, queue, use_usm_allocations };
    EXPECT_TRUE(ptr4.is_padded());

    // construct device_ptr
    const device_ptr_type ptr5{ plssvm::shape{ 42, 16 }, plssvm::shape{ 0, 0 }, queue, use_usm_allocations };
    EXPECT_FALSE(ptr5.is_padded());

    // construct empty device_ptr
    const device_ptr_type ptr6{};
    EXPECT_FALSE(ptr6.is_padded());
}

TYPED_TEST_P(DevicePtr, memset) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // memset values to all ones
    ptr.memset(1, 2);

    // copy values to host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<value_type> correct(ptr.size());
    std::memset(correct.data() + 2, 1, (correct.size() - 2) * sizeof(value_type));
    EXPECT_EQ(result, correct);
}

TYPED_TEST_P(DevicePtr, memset_with_numbytes) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // memset values to all ones
    ptr.memset(1, 2, 4 * sizeof(value_type));

    // copy values to host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<value_type> correct(ptr.size());
    std::memset(correct.data() + 2, 1, 4 * sizeof(value_type));
    EXPECT_EQ(result, correct);
}

TYPED_TEST_P(DevicePtr, memset_invalid_pos) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // perform invalid memset
    EXPECT_THROW_WHAT(ptr.memset(0, 10, 1),
                      plssvm::exception,
                      "Illegal access in memset!: 10 >= 10");
}

TYPED_TEST_P(DevicePtr, fill) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // fill values with a specific value
    ptr.fill(value_type{ 42.0 }, 2);

    // copy values to host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<value_type> correct(ptr.size(), value_type{ 42.0 });
    correct[0] = correct[1] = value_type{ 0.0 };
    EXPECT_EQ(result, correct);
}

TYPED_TEST_P(DevicePtr, fill_with_count) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // fill values with a specific value
    ptr.fill(value_type{ 42.0 }, 2, 4);

    // copy values to host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<value_type> correct(ptr.size(), value_type{ 0.0 });
    correct[2] = correct[3] = correct[4] = correct[5] = value_type{ 42.0 };
    EXPECT_EQ(result, correct);
}

TYPED_TEST_P(DevicePtr, fill_invalid_pos) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // perform invalid fill
    EXPECT_THROW_WHAT(ptr.fill(value_type{ 42.0 }, 10, 1),
                      plssvm::exception,
                      "Illegal access in fill!: 10 >= 10");
}

TYPED_TEST_P(DevicePtr, copy_vector) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values for correctness
    EXPECT_EQ(result, std::vector<value_type>(10, 42));
}

TYPED_TEST_P(DevicePtr, copy_vector_with_count_copy_back_all) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data, 1, 3);

    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);
    // check values for correctness
    EXPECT_EQ(result, (std::vector<value_type>{ value_type{ 0.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 0.0 }, value_type{ 0.0 } }));
}

TYPED_TEST_P(DevicePtr, copy_vector_with_count_copy_back_some) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data, 1, 3);

    // copy only a view values back from the device
    std::vector<value_type> result(3);
    ptr.copy_to_host(result, 2, 3);
    // check values for correctness
    EXPECT_EQ(result, (std::vector<value_type>{ value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 0.0 } }));
}

TYPED_TEST_P(DevicePtr, copy_vector_with_count_copy_to_too_many) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data, 2, 6);

    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);
    // check values for correctness
    EXPECT_EQ(result, (std::vector<value_type>{ value_type{ 0.0 }, value_type{ 0.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 42.0 } }));
}

TYPED_TEST_P(DevicePtr, copy_vector_too_few_host_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // try copying data to the device with too few elements
    std::vector<value_type> data(8, 42);
    EXPECT_THROW_WHAT(ptr.copy_to_device(data), plssvm::gpu_device_ptr_exception, "Too few data to perform copy (needed: 10, provided: 8)!");
}

TYPED_TEST_P(DevicePtr, copy_vector_too_few_buffer_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // try copying data back to the host with a buffer with too few elements
    std::vector<value_type> buffer(8);
    EXPECT_THROW_WHAT(ptr.copy_to_host(buffer), plssvm::gpu_device_ptr_exception, "Buffer too small to perform copy (needed: 10, provided: 8)!");
}

TYPED_TEST_P(DevicePtr, copy_vector_with_count_too_few_host_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // try copying data to the device with too few elements
    std::vector<value_type> data(4, 42);
    EXPECT_THROW_WHAT(ptr.copy_to_device(data, 1, 7), plssvm::gpu_device_ptr_exception, "Too few data to perform copy (needed: 7, provided: 4)!");
}

TYPED_TEST_P(DevicePtr, copy_vector_with_count_too_few_buffer_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // try copying data back to the host with a buffer with too few elements
    std::vector<value_type> buffer(4);
    EXPECT_THROW_WHAT(ptr.copy_to_host(buffer, 1, 7), plssvm::gpu_device_ptr_exception, "Buffer too small to perform copy (needed: 5, provided: 4)!");
}

TYPED_TEST_P(DevicePtr, copy_vector_strided) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 2 }, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(20);  // 5 x 4
    std::iota(data.begin(), data.end(), value_type{ 0.0 });

    // copy data to the device
    ptr.copy_to_device_strided(data, 5, 2, 2);
    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values for correctness
    const std::vector<value_type> correct_result = { 0.0, 1.0, 5.0, 6.0 };
    EXPECT_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtr, copy_vector_strided_invalid_spitch_width_combination) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 2 }, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(20);  // 5 x 4

    // copy data to the device
    EXPECT_THROW_WHAT(ptr.copy_to_device_strided(data, 5, 6, 2), plssvm::gpu_device_ptr_exception, "Invalid width and spitch combination specified (width: 6 <= spitch: 5)!");
}

TYPED_TEST_P(DevicePtr, copy_vector_strided_submatrix_too_big) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 2 }, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(20);  // 5 x 4

    // copy data to the device
    EXPECT_THROW_WHAT(ptr.copy_to_device_strided(data, 5, 5, 5), plssvm::gpu_device_ptr_exception, "The sub-matrix (5x5) to copy is to big (20)!");
}

TYPED_TEST_P(DevicePtr, copy_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data.data());
    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result.data());

    // check values for correctness
    EXPECT_EQ(result, std::vector<value_type>(10, 42));
}

TYPED_TEST_P(DevicePtr, copy_ptr_with_count_copy_back_all) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data.data(), 1, 3);

    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result.data());
    // check values for correctness
    EXPECT_EQ(result, (std::vector<value_type>{ value_type{ 0.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 0.0 }, value_type{ 0.0 } }));
}

TYPED_TEST_P(DevicePtr, copy_ptr_with_count_copy_back_some) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data.data(), 1, 3);

    // copy only a view values back from the device
    std::vector<value_type> result(3);
    ptr.copy_to_host(result.data(), 2, 3);
    // check values for correctness
    EXPECT_EQ(result, (std::vector<value_type>{ value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 0.0 } }));
}

TYPED_TEST_P(DevicePtr, copy_ptr_with_count_copy_to_too_many) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 6, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data.data(), 2, 6);

    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result.data());
    // check values for correctness
    EXPECT_EQ(result, (std::vector<value_type>{ value_type{ 0.0 }, value_type{ 0.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 42.0 }, value_type{ 42.0 } }));
}

TYPED_TEST_P(DevicePtr, copy_ptr_strided) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 2 }, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(20);  // 5 x 4
    std::iota(data.begin(), data.end(), value_type{ 0.0 });

    // copy data to the device
    ptr.copy_to_device_strided(data.data(), 5, 2, 2);
    // copy data back to the host
    std::vector<value_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values for correctness
    const std::vector<value_type> correct_result = { 0.0, 1.0, 5.0, 6.0 };
    EXPECT_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtr, copy_ptr_strided_invalid_spitch_width_combination) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 2 }, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(20);  // 5 x 4

    // copy data to the device
    EXPECT_THROW_WHAT(ptr.copy_to_device_strided(data.data(), 5, 6, 2), plssvm::exception, "Invalid width and spitch combination specified (width: 6 <= spitch: 5)!");
}

TYPED_TEST_P(DevicePtr, copy_device_ptr_to_other_device) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data);

    // other device_ptr
    device_ptr_type other_ptr{ 10, queue, !use_usm_allocations };
    ptr.copy_to_other_device(other_ptr);

    // copy data back to the host
    std::vector<value_type> result(other_ptr.size());
    other_ptr.copy_to_host(result);

    // check values for correctness
    EXPECT_EQ(result, std::vector<value_type>(10, 42));
}

TYPED_TEST_P(DevicePtr, copy_device_ptr_to_other_device_too_few_device_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data);

    // other device_ptr
    device_ptr_type other_ptr{ 5, queue, !use_usm_allocations };
    EXPECT_THROW_WHAT(ptr.copy_to_other_device(other_ptr), plssvm::exception, "Buffer too small to perform copy (needed: 10, provided: 5)!");
}

TYPED_TEST_P(DevicePtr, copy_device_ptr_to_other_device_with_count) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    std::vector<value_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data);

    // other device_ptr
    device_ptr_type other_ptr{ 5, queue, !use_usm_allocations };
    ptr.copy_to_other_device(other_ptr, 1, 5);

    // copy data back to the host
    std::vector<value_type> result(other_ptr.size());
    other_ptr.copy_to_host(result);

    // check values for correctness
    EXPECT_EQ(result, std::vector<value_type>(5, 42));
}

REGISTER_TYPED_TEST_SUITE_P(DevicePtr,
                            default_construct,
                            construct_size,
                            construct_shape,
                            construct_shape_and_padding,
                            move_construct,
                            move_construct_with_padding,
                            move_assign,
                            move_assign_with_padding,
                            swap_member_function,
                            swap_member_function_with_padding,
                            swap_free_function,
                            swap_free_function_with_padding,
                            operator_bool,
                            size,
                            shape,
                            empty,
                            padding,
                            size_padded,
                            shape_padded,
                            is_padded,
                            memset,
                            memset_with_numbytes,
                            memset_invalid_pos,
                            fill,
                            fill_with_count,
                            fill_invalid_pos,
                            copy_vector,
                            copy_vector_with_count_copy_back_all,
                            copy_vector_with_count_copy_back_some,
                            copy_vector_with_count_copy_to_too_many,
                            copy_vector_too_few_host_elements,
                            copy_vector_too_few_buffer_elements,
                            copy_vector_with_count_too_few_host_elements,
                            copy_vector_with_count_too_few_buffer_elements,
                            copy_vector_strided,
                            copy_vector_strided_invalid_spitch_width_combination,
                            copy_vector_strided_submatrix_too_big,
                            copy_ptr,
                            copy_ptr_with_count_copy_back_all,
                            copy_ptr_with_count_copy_back_some,
                            copy_ptr_with_count_copy_to_too_many,
                            copy_ptr_strided,
                            copy_ptr_strided_invalid_spitch_width_combination,
                            copy_device_ptr_to_other_device,
                            copy_device_ptr_to_other_device_too_few_device_elements,
                            copy_device_ptr_to_other_device_with_count);

template <typename T>
class DevicePtrLayout : public DevicePtr<T> {
  protected:
    using typename DevicePtr<T>::fixture_test_type;
};

TYPED_TEST_SUITE_P(DevicePtrLayout);

TYPED_TEST_P(DevicePtrLayout, copy_matrix) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    const plssvm::matrix<value_type, layout> data{ plssvm::shape{ 5, 3 }, value_type{ 42 } };

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ plssvm::shape{ 5, 3 }, value_type{ 0 } };
    ptr.copy_to_host(result);

    // check values for correctness
    plssvm::matrix<value_type, layout> correct_result{};
    switch (layout) {
        case plssvm::layout_type::aos:
            correct_result = plssvm::matrix<value_type, layout>{ { { 42, 42, 42 }, { 42, 42, 42 }, { 42, 42, 42 }, { 42, 0, 0 }, { 0, 0, 0 } } };
            break;
        case plssvm::layout_type::soa:
            correct_result = plssvm::matrix<value_type, layout>{ { { 42, 42, 0 }, { 42, 42, 0 }, { 42, 42, 0 }, { 42, 42, 0 }, { 42, 42, 0 } } };
            break;
    }
    EXPECT_FLOATING_POINT_MATRIX_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 5, 3 }, plssvm::shape{ 4, 4 }, queue, use_usm_allocations };

    // create data to copy to the device
    const plssvm::matrix<value_type, layout> data{ plssvm::shape{ 5, 3 }, value_type{ 42 }, plssvm::shape{ 4, 4 } };

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ plssvm::shape{ 5, 3 }, value_type{ 0 }, plssvm::shape{ 4, 4 } };
    ptr.copy_to_host(result);

    // check values for correctness
    EXPECT_FLOATING_POINT_MATRIX_EQ(result, data);
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_different_layouts) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::layout_type other_layout = layout == plssvm::layout_type::aos ? plssvm::layout_type::soa : plssvm::layout_type::aos;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // create data to copy to the device
    const plssvm::matrix<value_type, layout> data{ plssvm::shape{ 5, 3 }, value_type{ 42 } };

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    plssvm::matrix<value_type, other_layout> result{ plssvm::shape{ 5, 3 }, value_type{ 0 } };
    ptr.copy_to_host(result);

    // check values for correctness
    plssvm::matrix<value_type, other_layout> correct_result{};
    switch (other_layout) {
        case plssvm::layout_type::aos:
            correct_result = plssvm::matrix<value_type, other_layout>{ { { 42, 42, 42 }, { 42, 42, 42 }, { 42, 42, 42 }, { 42, 0, 0 }, { 0, 0, 0 } } };
            break;
        case plssvm::layout_type::soa:
            correct_result = plssvm::matrix<value_type, other_layout>{ { { 42, 42, 0 }, { 42, 42, 0 }, { 42, 42, 0 }, { 42, 42, 0 }, { 42, 42, 0 } } };
            break;
    }
    EXPECT_FLOATING_POINT_MATRIX_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_too_few_host_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // try copying data to the device with too few elements
    plssvm::matrix<value_type, layout> data{ plssvm::shape{ 2, 4 }, value_type{ 42 } };
    EXPECT_THROW_WHAT(ptr.copy_to_device(data), plssvm::gpu_device_ptr_exception, "Too few data to perform copy (needed: 10, provided: 8)!");
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_too_few_buffer_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // try copying data back to the host with a buffer with too few elements
    plssvm::matrix<value_type, layout> buffer{ plssvm::shape{ 2, 4 } };
    EXPECT_THROW_WHAT(ptr.copy_to_host(buffer), plssvm::gpu_device_ptr_exception, "Buffer too small to perform copy (needed: 10, provided: 8)!");
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_strided) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape(2, 3), queue, use_usm_allocations };

    // create data to copy to the device
    const auto data = util::generate_specific_matrix<plssvm::matrix<value_type, layout>>(plssvm::shape{ 5, 3 });

    // copy data to the device (strided)
    ptr.copy_to_device_strided(data, 1, 2);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ plssvm::shape{ 2, 3 }, value_type{ 0 } };
    ptr.copy_to_host(result);

    // check values for correctness
    plssvm::matrix<value_type, layout> correct_result{ { { 1.1, 1.2, 1.3 }, { 2.1, 2.2, 2.3 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_strided_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 3, 3 }, plssvm::shape{ 4, 4 }, queue, use_usm_allocations };

    // create data to copy to the device
    const auto data = util::generate_specific_matrix<plssvm::matrix<value_type, layout>>(plssvm::shape{ 5, 3 }, plssvm::shape{ 4, 4 });

    // copy data to the device
    ptr.copy_to_device_strided(data, 2, 3);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ plssvm::shape{ 3, 3 }, value_type{ 0 }, plssvm::shape{ 4, 4 } };
    ptr.copy_to_host(result);

    // check values for correctness
    plssvm::matrix<value_type, layout> correct_result{ { { 2.1, 2.2, 2.3 },
                                                         { 3.1, 3.2, 3.3 },
                                                         { 4.1, 4.2, 4.3 } },
                                                       plssvm::shape{ 4, 4 } };

    EXPECT_FLOATING_POINT_MATRIX_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_strided_different_layouts) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::layout_type other_layout = layout == plssvm::layout_type::aos ? plssvm::layout_type::soa : plssvm::layout_type::aos;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 3 }, queue, use_usm_allocations };

    // create data to copy to the device
    const auto data = util::generate_specific_matrix<plssvm::matrix<value_type, layout>>(plssvm::shape{ 5, 3 });

    // copy data to the device
    ptr.copy_to_device_strided(data, 0, 2);
    // copy data back to the host
    plssvm::matrix<value_type, other_layout> result{ plssvm::shape{ 2, 3 }, value_type{ 0 } };
    ptr.copy_to_host(result);

    // check values for correctness
    plssvm::matrix<value_type, other_layout> correct_result{};
    switch (other_layout) {
        case plssvm::layout_type::aos:
            correct_result = plssvm::matrix<value_type, other_layout>{ { { 0.1, 1.1, 0.2 }, { 1.2, 0.3, 1.3 } } };
            break;
        case plssvm::layout_type::soa:
            correct_result = plssvm::matrix<value_type, other_layout>{ { { 0.1, 0.3, 1.2 }, { 0.2, 1.1, 1.3 } } };
            break;
    }

    EXPECT_FLOATING_POINT_MATRIX_EQ(result, correct_result);
}

TYPED_TEST_P(DevicePtrLayout, copy_full_matrix_strided) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape(5, 3), queue, use_usm_allocations };

    // create data to copy to the device
    const auto data = util::generate_specific_matrix<plssvm::matrix<value_type, layout>>(plssvm::shape{ 5, 3 });

    // copy data to the device (strided)
    ptr.copy_to_device_strided(data, 0, 5);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ plssvm::shape{ 5, 3 }, value_type{ 0 } };
    ptr.copy_to_host(result);

    // check values for correctness
    EXPECT_FLOATING_POINT_MATRIX_EQ(result, data);
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_strided_too_few_host_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 5 }, queue, use_usm_allocations };

    // try copying data to the device with too few elements
    plssvm::matrix<value_type, layout> data{ plssvm::shape{ 2, 4 }, value_type{ 42 } };
    EXPECT_THROW_WHAT(ptr.copy_to_device_strided(data, 1, 1), plssvm::gpu_device_ptr_exception, "Too few data to perform copy (needed: 10, provided: 4)!");
}

TYPED_TEST_P(DevicePtrLayout, copy_matrix_strided_invalid_submatrix) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ plssvm::shape{ 2, 5 }, queue, use_usm_allocations };

    // try copying data to the device with too few elements
    plssvm::matrix<value_type, layout> data{ plssvm::shape{ 4, 5 }, value_type{ 42 } };
    EXPECT_THROW_WHAT(ptr.copy_to_device_strided(data, 4, 1), plssvm::gpu_device_ptr_exception, "Tried to copy lines 4-4 (zero-based index) to the device, but the matrix has only 4 lines!");
    EXPECT_THROW_WHAT(ptr.copy_to_device_strided(data, 3, 3), plssvm::gpu_device_ptr_exception, "Tried to copy lines 3-5 (zero-based index) to the device, but the matrix has only 4 lines!");
}

REGISTER_TYPED_TEST_SUITE_P(DevicePtrLayout,
                            copy_matrix,
                            copy_matrix_with_padding,
                            copy_matrix_different_layouts,
                            copy_matrix_too_few_host_elements,
                            copy_matrix_too_few_buffer_elements,
                            copy_matrix_strided,
                            copy_matrix_strided_with_padding,
                            copy_matrix_strided_different_layouts,
                            copy_full_matrix_strided,
                            copy_matrix_strided_too_few_host_elements,
                            copy_matrix_strided_invalid_submatrix);

template <typename T>
class DevicePtrDeathTest : public DevicePtr<T> { };

TYPED_TEST_SUITE_P(DevicePtrDeathTest);

TYPED_TEST_P(DevicePtrDeathTest, memset) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;

    // default construct device_ptr
    device_ptr_type ptr{};

    // memset values to all ones
    EXPECT_DEATH(ptr.memset(1, 2), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(ptr.memset(1, 2, 4), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
}

TYPED_TEST_P(DevicePtrDeathTest, fill) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;

    // default construct device_ptr
    device_ptr_type ptr{};

    // memset values to all ones
    EXPECT_DEATH(ptr.fill(value_type{ 1.0 }, 2), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(ptr.fill(value_type{ 1.0 }, 2, 4), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_invalid_host_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // copy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device(nullptr), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
    EXPECT_DEATH(ptr.copy_to_host(nullptr), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_invalid_device_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;

    // construct default device_ptr
    device_ptr_type def{};

    // copy with invalid device pointer
    std::vector<value_type> data(def.size());
    plssvm::aos_matrix<value_type> matr{};
    EXPECT_DEATH(def.copy_to_device(matr.data()), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_host(matr.data()), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_device(data.data()), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_host(data.data()), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_device(data), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_host(data), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_with_count_invalid_host_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // copy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device(nullptr, 0, 10), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
    EXPECT_DEATH(ptr.copy_to_host(nullptr, 0, 10), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_with_count_invalid_device_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;

    // construct default device_ptr
    device_ptr_type def{};

    // copy with invalid device pointer
    std::vector<value_type> data(def.size());
    EXPECT_DEATH(def.copy_to_device(data.data(), 0, 0), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_host(data.data(), 0, 0), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_device(data, 0, 0), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_host(data, 0, 0), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_strided_invalid_host_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct device_ptr
    device_ptr_type ptr{ 1, queue, use_usm_allocations };

    // copy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device_strided(nullptr, 0, 0, 0), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_strided_invalid_device_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;

    // construct default device_ptr
    device_ptr_type def{};

    // copy with invalid device pointer
    std::vector<value_type> data(def.size());
    EXPECT_DEATH(def.copy_to_device_strided(data.data(), 0, 0, 0), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(def.copy_to_device_strided(data, 0, 0, 0), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_to_other_device_invalid_device_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct default device_ptr
    device_ptr_type def{};
    device_ptr_type ptr{ 2, queue, use_usm_allocations };

    // copy with invalid device pointer
    EXPECT_DEATH(def.copy_to_other_device(ptr), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(ptr.copy_to_other_device(def), ::testing::HasSubstr("Invalid target pointer! Maybe target has been default constructed?"));
}

TYPED_TEST_P(DevicePtrDeathTest, copy_to_other_device_with_count_invalid_device_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr bool use_usm_allocations = test_type::use_usm_allocations;

    // construct default device_ptr
    device_ptr_type def{};
    device_ptr_type ptr{ 10, queue, use_usm_allocations };

    // copy with invalid device pointer
    EXPECT_DEATH(def.copy_to_other_device(ptr, 0, 10), ::testing::HasSubstr("Invalid data pointer! Maybe *this has been default constructed?"));
    EXPECT_DEATH(ptr.copy_to_other_device(def, 0, 10), ::testing::HasSubstr("Invalid target pointer! Maybe target has been default constructed?"));
}

REGISTER_TYPED_TEST_SUITE_P(DevicePtrDeathTest,
                            memset,
                            fill,
                            copy_invalid_host_ptr,
                            copy_invalid_device_ptr,
                            copy_with_count_invalid_host_ptr,
                            copy_with_count_invalid_device_ptr,
                            copy_strided_invalid_host_ptr,
                            copy_strided_invalid_device_ptr,
                            copy_to_other_device_invalid_device_ptr,
                            copy_to_other_device_with_count_invalid_device_ptr);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_DEVICE_PTR_TESTS_HPP_

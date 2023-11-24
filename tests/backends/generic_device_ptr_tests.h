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

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_MATRIX_EQ
#include "types_to_test.hpp"       // util::test_parameter_type_at_t

#include "gtest//gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EQ, EXPECT_NE, EXPECT_DEATH
                           // ::testing::{Test, hasSubstr}

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

    // default construct device_ptr
    const device_ptr_type ptr{};

    // empty data
    EXPECT_FALSE(static_cast<bool>(ptr));
    EXPECT_EQ(ptr.get(), nullptr);
    EXPECT_EQ(ptr.size(), 0);
    EXPECT_EQ(ptr.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(ptr.empty());
}
TYPED_TEST_P(DevicePtr, construct) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr{ 42, queue };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_EQ(ptr.size(), 42);
    EXPECT_EQ(ptr.extents(), (std::array<std::size_t, 2>{ 42, 0 }));
    EXPECT_FALSE(ptr.empty());
    // check padding
    EXPECT_EQ(ptr.padding(0), 0);
    EXPECT_EQ(ptr.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, construct_extents) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr{ { 42, 16 }, queue };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_EQ(ptr.size(), 42 * 16);
    EXPECT_EQ(ptr.extents(), (std::array<std::size_t, 2>{ 42, 16 }));
    EXPECT_FALSE(ptr.empty());
    // check padding
    EXPECT_EQ(ptr.padding(0), 0);
    EXPECT_EQ(ptr.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, construct_extents_and_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr{ { 42, 16 }, { 4, 4 }, queue };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_EQ(ptr.size(), (42 + 4) * (16 + 4));
    EXPECT_EQ(ptr.extents(), (std::array<std::size_t, 2>{ 42, 16 }));
    EXPECT_FALSE(ptr.empty());
    // check padding
    EXPECT_EQ(ptr.padding(0), 4);
    EXPECT_EQ(ptr.padding(1), 4);
}
TYPED_TEST_P(DevicePtr, move_construct) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type first{ 42, queue };
    const device_ptr_type second{ std::move(first) };

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    // EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 0 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 0);
    EXPECT_EQ(second.padding(1), 0);

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, move_construct_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type first{ { 42, 10 }, { 4, 5 }, queue };
    const device_ptr_type second{ std::move(first) };

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    // EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), (42 + 4) * (10 + 5));
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 10 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 4);
    EXPECT_EQ(second.padding(1), 5);

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, move_assign) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type first{ 42, queue };
    device_ptr_type second;

    // move assign
    second = std::move(first);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 0 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 0);
    EXPECT_EQ(second.padding(1), 0);

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, move_assign_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type first{ { 42, 10 }, { 4, 5 }, queue };
    device_ptr_type second;

    // move assign
    second = std::move(first);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), (42 + 4) * (10 + 5));
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 10 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 4);
    EXPECT_EQ(second.padding(1), 5);

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}

TYPED_TEST_P(DevicePtr, swap_member_function) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct two device_ptr
    device_ptr_type first{ 42, queue };
    device_ptr_type second{};

    // swap both device_ptr using the member function
    first.swap(second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 0 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 0);
    EXPECT_EQ(second.padding(1), 0);

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, swap_member_function_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct two device_ptr
    device_ptr_type first{ { 42, 10 }, { 4, 5 }, queue };
    device_ptr_type second{};

    // swap both device_ptr using the member function
    first.swap(second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), (42 + 4) * (10 + 5));
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 10 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 4);
    EXPECT_EQ(second.padding(1), 5);

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}
TYPED_TEST_P(DevicePtr, swap_free_function) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct two device_ptr
    device_ptr_type first{ 42, queue };
    device_ptr_type second;

    // swap both device_ptr using the free function
    using std::swap;
    swap(first, second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 0 }));
    EXPECT_FALSE(second.empty());

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
}
TYPED_TEST_P(DevicePtr, swap_free_function_with_padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct two device_ptr
    device_ptr_type first{ { 42, 10 }, { 4, 5 }, queue };
    device_ptr_type second;

    // swap both device_ptr using the free function
    using std::swap;
    swap(first, second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), (42 + 4) * (10 + 5));
    EXPECT_EQ(second.extents(), (std::array<std::size_t, 2>{ 42, 10 }));
    EXPECT_FALSE(second.empty());
    // check padding
    EXPECT_EQ(second.padding(0), 4);
    EXPECT_EQ(second.padding(1), 5);

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_EQ(first.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
    EXPECT_TRUE(first.empty());
    // check padding
    EXPECT_EQ(first.padding(0), 0);
    EXPECT_EQ(first.padding(1), 0);
}

TYPED_TEST_P(DevicePtr, operator_bool) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue };
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

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue };
    EXPECT_EQ(ptr1.size(), 42);

    // construct device_ptr with extents
    const device_ptr_type ptr2{ { 42, 16 }, queue };
    EXPECT_EQ(ptr2.size(), 42 * 16);

    // construct device_ptr with extents and padding
    const device_ptr_type ptr3{ { 42, 16 }, { 3, 3 }, queue };
    EXPECT_EQ(ptr3.size(), (42 + 3) * (16 + 3));

    // construct empty device_ptr
    const device_ptr_type ptr4{};
    EXPECT_EQ(ptr4.size(), 0);
}
TYPED_TEST_P(DevicePtr, size_idx) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue };
    EXPECT_EQ(ptr1.size(0), 42);
    EXPECT_EQ(ptr1.size(1), 0);

    // construct device_ptr with extents
    const device_ptr_type ptr2{ { 42, 16 }, queue };
    EXPECT_EQ(ptr2.size(0), 42);
    EXPECT_EQ(ptr2.size(1), 16);

    // construct empty device_ptr
    const device_ptr_type ptr3{};
    EXPECT_EQ(ptr3.size(0), 0);
    EXPECT_EQ(ptr3.size(1), 0);
}
TYPED_TEST_P(DevicePtr, extent) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue };
    EXPECT_EQ(ptr1.extents(), (std::array<std::size_t, 2>{ 42, 0 }));

    // construct device_ptr with extents
    const device_ptr_type ptr2{ { 42, 16 }, queue };
    EXPECT_EQ(ptr2.extents(), (std::array<std::size_t, 2>{ 42, 16 }));

    // construct empty device_ptr
    const device_ptr_type ptr3{};
    EXPECT_EQ(ptr3.extents(), (std::array<std::size_t, 2>{ 0, 0 }));
}
TYPED_TEST_P(DevicePtr, padding) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr{ { 42, 16 }, { 4, 5 }, queue };
    EXPECT_EQ(ptr.padding(0), 4);
    EXPECT_EQ(ptr.padding(1), 5);
}
TYPED_TEST_P(DevicePtr, empty) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    const device_ptr_type ptr1{ 42, queue };
    EXPECT_FALSE(ptr1.empty());

    // construct device_ptr
    const device_ptr_type ptr2{ { 42, 16 }, queue };
    EXPECT_FALSE(ptr2.empty());

    // construct device_ptr
    const device_ptr_type ptr3{ { 42, 16 }, { 3, 3 }, queue };
    EXPECT_FALSE(ptr3.empty());

    // construct empty device_ptr
    const device_ptr_type ptr4{};
    EXPECT_TRUE(ptr4.empty());
}

TYPED_TEST_P(DevicePtr, memset) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };

    // try copying data back to the host with a buffer with too few elements
    std::vector<value_type> buffer(4);
    EXPECT_THROW_WHAT(ptr.copy_to_host(buffer, 1, 7), plssvm::gpu_device_ptr_exception, "Buffer too small to perform copy (needed: 5, provided: 4)!");
}

TYPED_TEST_P(DevicePtr, copy_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };
    ptr.memset(0);

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

    // construct device_ptr
    device_ptr_type ptr{ 6, queue };
    ptr.memset(0);

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

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(DevicePtr,
                            default_construct, construct, construct_extents, construct_extents_and_padding,
                            move_construct, move_construct_with_padding, move_assign, move_assign_with_padding,
                            swap_member_function, swap_member_function_with_padding, swap_free_function, swap_free_function_with_padding,
                            operator_bool, size, size_idx, extent, padding, empty,
                            memset, memset_with_numbytes, memset_invalid_pos,
                            fill, fill_with_count, fill_invalid_pos,
                            copy_vector, copy_vector_with_count_copy_back_all, copy_vector_with_count_copy_back_some, copy_vector_with_count_copy_to_too_many,
                            copy_vector_too_few_host_elements, copy_vector_too_few_buffer_elements, copy_vector_with_count_too_few_host_elements, copy_vector_with_count_too_few_buffer_elements,
                            copy_ptr, copy_ptr_with_count_copy_back_all, copy_ptr_with_count_copy_back_some, copy_ptr_with_count_copy_to_too_many);
// clang-format on

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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

    // create data to copy to the device
    plssvm::matrix<value_type, layout> data{ 5, 3, value_type{ 42 } };

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ 5, 3, value_type{ 0 } };
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

    // construct device_ptr
    device_ptr_type ptr{ { 5, 3 }, { 4, 4 }, queue };
    ptr.memset(0);

    // create data to copy to the device
    plssvm::matrix<value_type, layout> data{ 5, 3, value_type{ 42 }, 4, 4 };

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    plssvm::matrix<value_type, layout> result{ 5, 3, value_type{ 0 }, 4, 4 };
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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };
    ptr.memset(0);

    // create data to copy to the device
    plssvm::matrix<value_type, layout> data{ 5, 3, value_type{ 42 } };

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    plssvm::matrix<value_type, other_layout> result{ 5, 3, value_type{ 0 } };
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

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

    // try copying data to the device with too few elements
    plssvm::matrix<value_type, layout> data{ 2, 4, value_type{ 42 } };
    EXPECT_THROW_WHAT(ptr.copy_to_device(data), plssvm::gpu_device_ptr_exception, "Too few data to perform copy (needed: 10, provided: 8)!");
}
TYPED_TEST_P(DevicePtrLayout, copy_matrix_too_few_buffer_elements) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using value_type = typename device_ptr_type::value_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();
    constexpr plssvm::layout_type layout = util::test_parameter_value_at_v<0, TypeParam>;

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

    // try copying data back to the host with a buffer with too few elements
    plssvm::matrix<value_type, layout> buffer{ 2, 4 };
    EXPECT_THROW_WHAT(ptr.copy_to_host(buffer), plssvm::gpu_device_ptr_exception, "Buffer too small to perform copy (needed: 10, provided: 8)!");
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(DevicePtrLayout,
                            copy_matrix, copy_matrix_with_padding, copy_matrix_different_layouts,
                            copy_matrix_too_few_host_elements, copy_matrix_too_few_buffer_elements);
// clang-format on

template <typename T>
class DevicePtrDeathTest : public DevicePtr<T> {};
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

TYPED_TEST_P(DevicePtrDeathTest, copy_ptr_invalid_host_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

    // copy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device(nullptr), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
    EXPECT_DEATH(ptr.copy_to_host(nullptr), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
}
TYPED_TEST_P(DevicePtrDeathTest, copy_ptr_invalid_device_ptr) {
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

TYPED_TEST_P(DevicePtrDeathTest, copy_ptr_with_count_invalid_host_ptr) {
    using test_type = typename TestFixture::fixture_test_type;
    using device_ptr_type = typename test_type::device_ptr_type;
    using queue_type = typename test_type::queue_type;
    const queue_type &queue = test_type::default_queue();

    // construct device_ptr
    device_ptr_type ptr{ 10, queue };

    // copy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device(nullptr, 0, 10), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
    EXPECT_DEATH(ptr.copy_to_host(nullptr, 0, 10), ::testing::HasSubstr("Invalid host pointer for the data to copy!"));
}
TYPED_TEST_P(DevicePtrDeathTest, copy_ptr_with_count_invalid_device_ptr) {
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

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(DevicePtrDeathTest,
                            memset, fill,
                            copy_ptr_invalid_host_ptr, copy_ptr_invalid_device_ptr,
                            copy_ptr_with_count_invalid_host_ptr, copy_ptr_with_count_invalid_device_ptr);
// clang-format on

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_DEVICE_PTR_TESTS_HPP_

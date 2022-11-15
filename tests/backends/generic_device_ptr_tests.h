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

#include "plssvm/backends/CUDA/exceptions.hpp"

#include "custom_test_macros.hpp"

#include <numeric>

#include "gtest//gtest.h"

namespace generic {

template <typename device_ptr_type>
void test_default_construct() {
    // default construct device_ptr
    const device_ptr_type ptr{};

    // empty data
    EXPECT_FALSE(static_cast<bool>(ptr));
    EXPECT_EQ(ptr.get(), nullptr);
    EXPECT_EQ(ptr.size(), 0);
    EXPECT_TRUE(ptr.empty());
}

template <typename device_ptr_type, typename queue_type>
void test_construct(queue_type queue) {
    // construct device_ptr
    const device_ptr_type ptr{ 42, queue };

    // check data
    EXPECT_TRUE(static_cast<bool>(ptr));
    EXPECT_EQ(ptr.queue(), queue);
    EXPECT_NE(ptr.get(), nullptr);
    EXPECT_EQ(ptr.size(), 42);
    EXPECT_FALSE(ptr.empty());
}

template <typename device_ptr_type, typename queue_type>
void test_move_construct(queue_type queue) {
    // construct device_ptr
    device_ptr_type first{ 42, queue };
    const device_ptr_type second{ std::move(first) };

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_FALSE(second.empty());

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_TRUE(first.empty());
}

template <typename device_ptr_type, typename queue_type>
void test_move_assign(queue_type queue) {
    // construct device_ptr
    device_ptr_type first{ 42, queue };
    device_ptr_type second;

    // move assign
    second = std::move(first);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_FALSE(second.empty());

    // check moved-from data
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_TRUE(first.empty());
}

template <typename device_ptr_type, typename queue_type>
void test_swap(queue_type queue) {
    // construct two device_ptr
    device_ptr_type first{ 42, queue };
    device_ptr_type second;

    // swap both device_ptr using the member function
    first.swap(second);

    // check data
    EXPECT_TRUE(static_cast<bool>(second));
    EXPECT_EQ(second.queue(), queue);
    EXPECT_NE(second.get(), nullptr);
    EXPECT_EQ(second.size(), 42);
    EXPECT_FALSE(second.empty());

    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(first.get(), nullptr);
    EXPECT_EQ(first.size(), 0);
    EXPECT_TRUE(first.empty());

    // swap them back using the free function
    using std::swap;
    swap(first, second);

    // check data
    EXPECT_TRUE(static_cast<bool>(first));
    EXPECT_EQ(first.queue(), queue);
    EXPECT_NE(first.get(), nullptr);
    EXPECT_EQ(first.size(), 42);
    EXPECT_FALSE(first.empty());

    EXPECT_FALSE(static_cast<bool>(second));
    EXPECT_EQ(second.get(), nullptr);
    EXPECT_EQ(second.size(), 0);
    EXPECT_TRUE(second.empty());
}

template <typename device_ptr_type>
void test_memset() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // memset values to all ones
    ptr.memset(1, 2);

    // copy values to host
    std::vector<real_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<real_type> correct(ptr.size());
    std::memset(correct.data() + 2, 1, (correct.size() - 2) * sizeof(real_type));
    EXPECT_EQ(result, correct);
}
template <typename device_ptr_type>
void test_memset_with_count() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // memset values to all ones
    ptr.memset(1, 2, 4 * sizeof(real_type));

    // copy values to host
    std::vector<real_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<real_type> correct(ptr.size());
    std::memset(correct.data() + 2, 1, 4 * sizeof(real_type));
    EXPECT_EQ(result, correct);
}
template <typename device_ptr_type>
void test_memset_death_test() {
    using real_type = typename device_ptr_type::value_type;

    // default construct device_ptr
    device_ptr_type ptr{};

    // memset values to all ones
    EXPECT_DEATH(ptr.memset(1, 2), "Invalid data pointer!");
    EXPECT_DEATH(ptr.memset(1, 2, 4), "Invalid data pointer!");
}

template <typename device_ptr_type>
void test_fill() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // memset values to all ones
    ptr.fill(real_type{ 42.0 }, 2);

    // copy values to host
    std::vector<real_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values
    std::vector<real_type> correct(ptr.size(), 42);
    correct[0] = real_type{ 0.0 };
    correct[1] = real_type{ 0.0 };
    EXPECT_EQ(result, correct);
}

template <typename device_ptr_type>
void test_copy_vector() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // create data to copy to the device
    std::vector<real_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data);
    // copy data back to the host
    std::vector<real_type> result(ptr.size());
    ptr.copy_to_host(result);

    // check values for correctness
    EXPECT_EQ(result, std::vector<real_type>(10, 42));
}
template <typename device_ptr_type>
void test_copy_vector_exception() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // try copying data to the device with too few elements
    std::vector<real_type> data(8, 42);
    EXPECT_THROW_WHAT(ptr.copy_to_device(data), plssvm::gpu_device_ptr_exception, "Too few data to perform memcpy (needed: 10, provided: 8)!");

    // try copying data back to the host with a buffer with too few elements
    std::vector<real_type> result(8);
    EXPECT_THROW_WHAT(ptr.copy_to_host(data), plssvm::gpu_device_ptr_exception, "Buffer too small to perform memcpy (needed: 10, provided: 8)!");
}

template <typename device_ptr_type>
void test_copy_vector_with_count() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 6 };

    // create data to copy to the device
    std::vector<real_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data, 1, 3);

    {
        // copy data back to the host
        std::vector<real_type> result(ptr.size());
        ptr.copy_to_host(result);
        // check values for correctness
        EXPECT_EQ(result, (std::vector<real_type>{ real_type{ 0.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 0.0 }, real_type{ 0.0 } }));
    }
    {
        // copy only a view values back from the device
        std::vector<real_type> result(3);
        ptr.copy_to_host(result, 2, 3);
        // check values for correctness
        EXPECT_EQ(result, (std::vector<real_type>{ real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 0.0 } }));
    }

    // copy data to the device
    ptr.copy_to_device(data, 2, 6);

    {
        // copy data back to the host
        std::vector<real_type> result(ptr.size());
        ptr.copy_to_host(result);
        // check values for correctness
        EXPECT_EQ(result, (std::vector<real_type>{ real_type{ 0.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 } }));
    }
}
template <typename device_ptr_type>
void test_copy_vector_with_count_exception() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 6 };

    // try copying data to the device with too few elements
    std::vector<real_type> data(4, 42);
    EXPECT_THROW_WHAT(ptr.copy_to_device(data, 1, 7), plssvm::gpu_device_ptr_exception, "Too few data to perform memcpy (needed: 5, provided: 4)!");

    // try copying data back to the host with a buffer with too few elements
    std::vector<real_type> result(4);
    EXPECT_THROW_WHAT(ptr.copy_to_host(data, 1, 7), plssvm::gpu_device_ptr_exception, "Buffer too small to perform memcpy (needed: 5, provided: 4)!");
}

template <typename device_ptr_type>
void test_copy_ptr() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // create data to copy to the device
    std::vector<real_type> data(14, 42);

    // copy data to the device
    ptr.copy_to_device(data.data());
    // copy data back to the host
    std::vector<real_type> result(ptr.size());
    ptr.copy_to_host(result.data());

    // check values for correctness
    EXPECT_EQ(result, std::vector<real_type>(10, 42));
}
template <typename device_ptr_type>
void test_copy_ptr_death_test() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // memcpy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device(nullptr), "Invalid pointer for the data to copy!");
    EXPECT_DEATH(ptr.copy_to_host(nullptr), "Invalid pointer for the data to copy!");

    // construct default device_ptr
    device_ptr_type def{};

    // memcpy with invalid data pointer
    std::vector<real_type> data(def.size());
    EXPECT_DEATH(def.copy_to_device(data), "Invalid data pointer!");
    EXPECT_DEATH(def.copy_to_host(data), "Invalid data pointer!");
}

template <typename device_ptr_type>
void test_copy_ptr_with_count() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 6 };

    // create data to copy to the device
    std::vector<real_type> data(6, 42);

    // copy data to the device
    ptr.copy_to_device(data.data(), 1, 3);

    {
        // copy data back to the host
        std::vector<real_type> result(ptr.size());
        ptr.copy_to_host(result.data());
        // check values for correctness
        EXPECT_EQ(result, (std::vector<real_type>{ real_type{ 0.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 0.0 }, real_type{ 0.0 } }));
    }
    {
        // copy only a view values back from the device
        std::vector<real_type> result(3);
        ptr.copy_to_host(result.data(), 2, 3);
        // check values for correctness
        EXPECT_EQ(result, (std::vector<real_type>{ real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 0.0 } }));
    }

    // copy data to the device
    ptr.copy_to_device(data.data(), 2, 6);

    {
        // copy data back to the host
        std::vector<real_type> result(ptr.size());
        ptr.copy_to_host(result.data());
        // check values for correctness
        EXPECT_EQ(result, (std::vector<real_type>{ real_type{ 0.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 }, real_type{ 42.0 } }));
    }
}
template <typename device_ptr_type>
void test_copy_ptr_with_count_death_test() {
    using real_type = typename device_ptr_type::value_type;

    // construct device_ptr
    device_ptr_type ptr{ 10 };

    // memcpy with invalid data pointer
    EXPECT_DEATH(ptr.copy_to_device(nullptr, 0, 10), "Invalid pointer for the data to copy!");
    EXPECT_DEATH(ptr.copy_to_host(nullptr, 0, 10), "Invalid pointer for the data to copy!");

    // construct default device_ptr
    device_ptr_type def{};

    // memcpy with invalid data pointer
    std::vector<real_type> data(def.size());
    EXPECT_DEATH(def.copy_to_device(data, 0, 0), "Invalid data pointer!");
    EXPECT_DEATH(def.copy_to_host(data, 0, 0), "Invalid data pointer!");
}

}  // namespace generic

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_DEVICE_PTR_TESTS_HPP_

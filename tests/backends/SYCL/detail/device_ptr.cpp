/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the OpenCL backend device pointer.
 */

#include "plssvm/backends/SYCL/detail/device_ptr.hpp"
#include "plssvm/backends/SYCL/detail/constants.hpp"  // forward declaration and namespace alias
#include "plssvm/target_platforms.hpp"

#include "plssvm/backends/SYCL/detail/queue_impl.hpp"

#include "backends/generic_device_ptr_tests.h"

#include "sycl/sycl.hpp"  // sycl::queue

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Types

#include <memory>  // std::unique_ptr

template <typename T>
struct device_ptr_test_type {
    using device_ptr_type = plssvm::sycl::detail::device_ptr<T>;
    using queue_type = plssvm::sycl::detail::queue;

    static const queue_type &default_queue() {
        static queue_type queue{ std::make_unique<plssvm::sycl::detail::queue::queue_impl>() };
//        static queue_type queue{ plssvm::sycl::detail::get_device_list(plssvm::target_platform::automatic).first.front() };
        return queue;
    }
};

using device_ptr_test_types = ::testing::Types<
    device_ptr_test_type<float>,
    device_ptr_test_type<double>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(SYCLBackend, DevicePtr, device_ptr_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(SYCLBackendDeathTest, DevicePtrDeathTest, device_ptr_test_types);

// TODO: linker error because of constructor call
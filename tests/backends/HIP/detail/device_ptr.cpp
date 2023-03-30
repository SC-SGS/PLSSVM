/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the HIP backend device pointer.
 */

#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"  // plssvm::hip::detail::device_ptr

#include "../../generic_device_ptr_tests.h"  // generic device pointer tests to instantiate

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Types

template <typename T>
struct device_ptr_test_type {
    using device_ptr_type = plssvm::hip::detail::device_ptr<T>;
    using queue_type = int;

    static const queue_type &default_queue() {
        static const queue_type queue = 0;
        return queue;
    }
};

using device_ptr_test_types = ::testing::Types<
    device_ptr_test_type<float>,
    device_ptr_test_type<double>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPBackend, DevicePtr, device_ptr_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPBackendDeathTest, DevicePtrDeathTest, device_ptr_test_types);
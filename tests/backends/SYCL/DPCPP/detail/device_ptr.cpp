/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the SYCL backend device pointer using DPC++ as SYCL implementation.
 */

#include "plssvm/backends/SYCL/DPCPP/detail/device_ptr.hpp"  // plssvm::dpcpp::detail::device_ptr
#include "backends/generic_device_ptr_tests.h"               // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"     // plssvm::dpcpp::detail::get_default_device

#include "gtest/gtest.h"                                     // INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Types

template <typename device_ptr_t>
struct dpcpp_device_ptr_test_type {
    using device_ptr_type = device_ptr_t;
    using queue_type = typename device_ptr_type::queue_type;

    static const queue_type &default_queue() {
        static const queue_type queue = plssvm::dpcpp::detail::get_default_queue();
        return queue;
    }
};

using dpcpp_device_ptr_test_types = ::testing::Types<
    dpcpp_device_ptr_test_type<plssvm::dpcpp::detail::device_ptr<float>>,
    dpcpp_device_ptr_test_type<plssvm::dpcpp::detail::device_ptr<double>>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtr, DevicePtr, dpcpp_device_ptr_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtrDeathTest, DevicePtrDeathTest, dpcpp_device_ptr_test_types);
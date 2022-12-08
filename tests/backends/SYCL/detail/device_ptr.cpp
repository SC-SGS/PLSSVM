/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the OpenCL backend device pointer.
 */

#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "backends/generic_device_ptr_tests.h"

#include "sycl/sycl.hpp"  // sycl::queue

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Types

#include <memory>  // std::unique_ptr

template <typename device_ptr_t, typename queue_t>
struct device_ptr_test_type {
    using device_ptr_type = device_ptr_t;
    using queue_type = queue_t;

    static const queue_type &default_queue() {
        // TODO: linker error because of constructor call
        static queue_type queue{ std::make_unique<typename queue_type::queue_impl>() };
        return queue;
    }
};

#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    #include "plssvm/backends/SYCL/DPCPP/detail/device_ptr.hpp"
    #include "plssvm/backends/SYCL/DPCPP/detail/queue_impl.hpp"

using dpcpp_device_ptr_test_types = ::testing::Types<
    device_ptr_test_type<plssvm::dpcpp::detail::device_ptr<float>, plssvm::dpcpp::detail::queue>,
    device_ptr_test_type<plssvm::dpcpp::detail::device_ptr<double>, plssvm::dpcpp::detail::queue>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPBackend, DevicePtr, dpcpp_device_ptr_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPBackendDeathTest, DevicePtrDeathTest, dpcpp_device_ptr_test_types);
#endif

#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    #include "plssvm/backends/SYCL/hipSYCL/detail/device_ptr.hpp"
    #include "plssvm/backends/SYCL/hipSYCL/detail/queue_impl.hpp"

using hipsycl_device_ptr_test_types = ::testing::Types<
    device_ptr_test_type<plssvm::hipsycl::detail::device_ptr<float>, plssvm::hipsycl::detail::queue>,
    device_ptr_test_type<plssvm::hipsycl::detail::device_ptr<double>, plssvm::hipsycl::detail::queue>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLBackend, DevicePtr, hipsycl_device_ptr_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLBackendDeathTest, DevicePtrDeathTest, hipsycl_device_ptr_test_types);
#endif
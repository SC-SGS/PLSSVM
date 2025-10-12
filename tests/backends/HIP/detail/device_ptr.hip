/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the HIP backend device pointer.
 */

#include "plssvm/backends/HIP/csvm.hpp"                   // plssvm::hip::csvc
#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"  // plssvm::hip::detail::device_ptr
#include "plssvm/backends/HIP/exceptions.hpp"             // plssvm::hip::backend_exception
#include "plssvm/shape.hpp"                               // plssvm::shape

#include "tests/backends/generic_device_ptr_tests.hpp"  // generic device pointer tests to instantiate
#include "tests/naming.hpp"                             // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                      // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list, real_type_gtest}

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T>
class HIPDevicePtrConstruct : public ::testing::Test { };

TYPED_TEST_SUITE(HIPDevicePtrConstruct, util::real_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(HIPDevicePtrConstruct, construct_invalid_queue) {
    using real_type = util::test_parameter_type_at_t<0, TypeParam>;

    // the number of devices
    const std::size_t num_devices = plssvm::hip::csvc{}.num_available_devices();

    EXPECT_THROW_WHAT_MATCHER((plssvm::hip::detail::device_ptr<real_type>(plssvm::shape{ 4, 4 }, -1)),
                              plssvm::hip::backend_exception,
                              ::testing::HasSubstr(fmt::format("Illegal device ID! Must be in range: [0, {}) but is -1.", num_devices)));

    EXPECT_THROW_WHAT_MATCHER((plssvm::hip::detail::device_ptr<real_type>(plssvm::shape{ 4, 4 }, num_devices)),
                              plssvm::hip::backend_exception,
                              ::testing::HasSubstr(fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", num_devices, num_devices)));
}

template <typename T>
struct hip_device_ptr_test_type {
    using device_ptr_type = plssvm::hip::detail::device_ptr<T>;
    using queue_type = int;

    static const queue_type &default_queue() {
        static const queue_type queue = 0;
        return queue;
    }
};

using hip_device_ptr_tuple = std::tuple<hip_device_ptr_test_type<float>, hip_device_ptr_test_type<double>>;

// the tests used in the instantiated GTest test suites
using hip_device_ptr_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<hip_device_ptr_tuple>>;
using hip_device_ptr_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<hip_device_ptr_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPDevicePtr, DevicePtr, hip_device_ptr_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPDevicePtr, DevicePtrLayout, hip_device_ptr_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(HIPDevicePtrDeathTest, DevicePtrDeathTest, hip_device_ptr_type_gtest, naming::test_parameter_to_name);

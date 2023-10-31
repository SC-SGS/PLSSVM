/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the SYCL backend device pointer using hipSYCL as SYCL implementation.
 */

#include "plssvm/backends/SYCL/hipSYCL/detail/device_ptr.hpp"  // plssvm::hipsycl::detail::device_ptr
#include "plssvm/backends/SYCL/hipSYCL/detail/utility.hpp"     // plssvm::hipsycl::detail::get_default_device

#include "../../../generic_device_ptr_tests.h"  // generic device pointer tests to instantiate

#include "../../../../naming.hpp"         // naming::test_parameter_to_name
#include "../../../../types_to_test.hpp"  // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T>
struct hipsycl_device_ptr_test_type {
    using device_ptr_type = plssvm::hipsycl::detail::device_ptr<T>;
    using queue_type = typename device_ptr_type::queue_type;

    static const queue_type &default_queue() {
        static const queue_type queue = plssvm::hipsycl::detail::get_default_queue();
        return queue;
    }
};
using device_ptr_test_types = std::tuple<hipsycl_device_ptr_test_type<float>, hipsycl_device_ptr_test_type<double>>;

// the tests used in the instantiated GTest test suites
using device_ptr_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<device_ptr_test_types>>;
using device_ptr_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<device_ptr_test_types>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLDevicePtr, DevicePtr, device_ptr_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLDevicePtr, DevicePtrLayout, device_ptr_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLDevicePtrDeathTest, DevicePtrDeathTest, device_ptr_type_gtest, naming::test_parameter_to_name);
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

#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"  // plssvm::dpcpp::detail::get_default_device

#include "tests/backends/generic_device_ptr_tests.hpp"  // generic device pointer tests to instantiate
#include "tests/naming.hpp"                             // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                      // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T, bool UUA>
struct dpcpp_device_ptr_test_type {
    using device_ptr_type = plssvm::dpcpp::detail::device_ptr<T>;
    using queue_type = typename device_ptr_type::queue_type;
    constexpr static bool use_usm_allocations = UUA;

    static const queue_type &default_queue() {
        static const queue_type queue = plssvm::dpcpp::detail::get_default_queue();
        return queue;
    }
};

using dpcpp_device_ptr_tuple = std::tuple<dpcpp_device_ptr_test_type<float, false>, dpcpp_device_ptr_test_type<double, false>>;

// the tests used in the instantiated GTest test suites
using dpcpp_device_ptr_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_device_ptr_tuple>>;
using dpcpp_device_ptr_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_device_ptr_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtr, DevicePtr, dpcpp_device_ptr_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtr, DevicePtrLayout, dpcpp_device_ptr_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtrDeathTest, DevicePtrDeathTest, dpcpp_device_ptr_type_gtest, naming::test_parameter_to_name);

//
// test USM pointer
//

using dpcpp_device_ptr_usm_tuple = std::tuple<dpcpp_device_ptr_test_type<float, true>, dpcpp_device_ptr_test_type<double, true>>;

// the tests used in the instantiated GTest test suites
using dpcpp_device_ptr_usm_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_device_ptr_usm_tuple>>;
using dpcpp_device_ptr_usm_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_device_ptr_usm_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtrUSM, DevicePtr, dpcpp_device_ptr_usm_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtrUSM, DevicePtrLayout, dpcpp_device_ptr_usm_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPDevicePtrUSMDeathTest, DevicePtrDeathTest, dpcpp_device_ptr_usm_type_gtest, naming::test_parameter_to_name);

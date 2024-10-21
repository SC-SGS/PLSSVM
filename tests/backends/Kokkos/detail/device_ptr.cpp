/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the Kokkos backend device pointer.
 */

#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"  // plssvm::kokkos::detail::device_ptr

#include "tests/backends/generic_device_ptr_tests.hpp"  // generic device pointer tests to instantiate
#include "tests/naming.hpp"                             // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                      // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T>
struct kokkos_device_ptr_test_type {
    using device_ptr_type = plssvm::kokkos::detail::device_ptr<T>;
    using queue_type = Kokkos::DefaultExecutionSpace;

    static const queue_type &default_queue() {
        static const queue_type queue{};
        return queue;
    }
};

using kokkos_device_ptr_tuple = std::tuple<kokkos_device_ptr_test_type<float>, kokkos_device_ptr_test_type<double>>;

// the tests used in the instantiated GTest test suites
using kokkos_device_ptr_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<kokkos_device_ptr_tuple>>;
using kokkos_device_ptr_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<kokkos_device_ptr_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosDevicePtr, DevicePtr, kokkos_device_ptr_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosDevicePtr, DevicePtrLayout, kokkos_device_ptr_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(KokkosDevicePtrDeathTest, DevicePtrDeathTest, kokkos_device_ptr_type_gtest, naming::test_parameter_to_name);

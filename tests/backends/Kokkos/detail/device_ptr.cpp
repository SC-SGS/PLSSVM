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

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"        // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_space.hpp"              // plssvm::kokkos::execution_space
#include "plssvm/backends/Kokkos/execution_space_type_traits.hpp"  // plssvm::kokkos::execution_space_to_kokkos_type_t

#include "tests/backends/generic_device_ptr_tests.hpp"  // generic device pointer tests to instantiate
#include "tests/backends/Kokkos/utility.hpp"            // util::create_kokkos_test_tuple_impl
#include "tests/naming.hpp"                             // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                      // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list},
                                                        // util::detail::concat_tuple_types_t

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>  // std::tuple

template <typename T, plssvm::kokkos::execution_space exec_space>
struct kokkos_device_ptr_test_type {
    using device_ptr_type = plssvm::kokkos::detail::device_ptr<T>;
    using queue_type = plssvm::kokkos::detail::device_wrapper;
    constexpr static plssvm::kokkos::execution_space space = exec_space;

    static const queue_type &default_queue() {
        static const queue_type queue{ plssvm::kokkos::execution_space_to_kokkos_type_t<space>{} };
        return queue;
    }
};

template <plssvm::kokkos::execution_space space>
using kokkos_device_ptr_test_type_float = kokkos_device_ptr_test_type<float, space>;
template <plssvm::kokkos::execution_space space>
using kokkos_device_ptr_test_type_double = kokkos_device_ptr_test_type<double, space>;

using kokkos_device_ptr_tuple = util::detail::concat_tuple_types_t<util::create_kokkos_test_tuple_t<kokkos_device_ptr_test_type_float>,
                                                                   util::create_kokkos_test_tuple_t<kokkos_device_ptr_test_type_double>>;

// the tests used in the instantiated GTest test suites
using kokkos_device_ptr_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<kokkos_device_ptr_tuple>>;
using kokkos_device_ptr_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<kokkos_device_ptr_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosDevicePtr, DevicePtr, kokkos_device_ptr_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosDevicePtr, DevicePtrLayout, kokkos_device_ptr_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(KokkosDevicePtrDeathTest, DevicePtrDeathTest, kokkos_device_ptr_type_gtest, naming::test_parameter_to_name);

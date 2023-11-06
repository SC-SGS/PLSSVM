/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the OpenCL backend device pointer.
 */

#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"     // plssvm::opencl::detail::device_ptr
#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/context.hpp"        // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // plssvm::opencl::detail::get_contexts

#include "backends/generic_device_ptr_tests.h"  // generic device pointer tests to instantiate
#include "naming.hpp"                           // naming::test_parameter_to_name
#include "types_to_test.hpp"                    // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, layout_type_list}

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <tuple>   // std::tuple
#include <vector>  // std::vector

template <typename T>
struct opencl_device_ptr_test_type {
    using device_ptr_type = plssvm::opencl::detail::device_ptr<T>;
    using queue_type = plssvm::opencl::detail::command_queue;

    static const queue_type &default_queue() {
        static const std::vector<plssvm::opencl::detail::context> contexts{ plssvm::opencl::detail::get_contexts(plssvm::target_platform::automatic).first };
        static const plssvm::opencl::detail::command_queue queue{ contexts[0], contexts[0].devices[0] };
        return queue;
    }
};
using opencl_device_ptr_tuple = std::tuple<opencl_device_ptr_test_type<float>, opencl_device_ptr_test_type<double>>;

// the tests used in the instantiated GTest test suites
using opencl_device_ptr_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<opencl_device_ptr_tuple>>;
using opencl_device_ptr_layout_type_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<opencl_device_ptr_tuple>, util::layout_type_list>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLDevicePtr, DevicePtr, opencl_device_ptr_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLDevicePtr, DevicePtrLayout, opencl_device_ptr_layout_type_gtest, naming::test_parameter_to_name);

INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLDevicePtrDeathTest, DevicePtrDeathTest, opencl_device_ptr_type_gtest, naming::test_parameter_to_name);
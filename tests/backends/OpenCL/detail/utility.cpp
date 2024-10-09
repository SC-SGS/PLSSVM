/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the OpenCL backend.
 */

#include "plssvm/backends/OpenCL/detail/utility.hpp"

#include "plssvm/backends/execution_range.hpp"              // plssvm::detail::dim_type
#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/context.hpp"        // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception

#include "CL/cl.h"  // CL_SUCCESS, CL_DEVICE_NOT_FOUND

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_NO_THROW, EXPECT_FALSE

#include <cstddef>  // std::size_t
#include <regex>    // std::regex, std::regex::extended, std::regex_match
#include <string>   // std::string
#include <vector>   // std::vector

TEST(OpenCLUtility, error_check) {
    // CL_SUCCESS must not throw
    EXPECT_NO_THROW(PLSSVM_OPENCL_ERROR_CHECK(CL_SUCCESS, "success!"));

    // any other code must throw
    EXPECT_THROW_WHAT(PLSSVM_OPENCL_ERROR_CHECK(CL_DEVICE_NOT_FOUND, "error"),
                      plssvm::opencl::backend_exception,
                      "OpenCL assert 'CL_DEVICE_NOT_FOUND' (-1): error!");
}

TEST(OpenCLUtility, dim_type_to_native_1) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a OpenCL std::vector<std::size_t>
    const std::vector<std::size_t> native_dim = plssvm::opencl::detail::dim_type_to_native<1>(dim);

    // check values for correctness
    ASSERT_EQ(native_dim.size(), 1);
    EXPECT_EQ(native_dim[0], dim.x);
}

TEST(OpenCLUtility, dim_type_to_native_2) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a OpenCL std::vector<std::size_t>
    const std::vector<std::size_t> native_dim = plssvm::opencl::detail::dim_type_to_native<2>(dim);

    // check values for correctness
    ASSERT_EQ(native_dim.size(), 2);
    EXPECT_EQ(native_dim[0], dim.x);
    EXPECT_EQ(native_dim[1], dim.y);
}

TEST(OpenCLUtility, dim_type_to_native_3) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a OpenCL std::vector<std::size_t>
    const std::vector<std::size_t> native_dim = plssvm::opencl::detail::dim_type_to_native<3>(dim);

    // check values for correctness
    ASSERT_EQ(native_dim.size(), 3);
    EXPECT_EQ(native_dim[0], dim.x);
    EXPECT_EQ(native_dim[1], dim.y);
    EXPECT_EQ(native_dim[2], dim.z);
}

TEST(OpenCLUtility, get_contexts) {
    const auto &[contexts, actual_target] = plssvm::opencl::detail::get_contexts(plssvm::target_platform::automatic);
    // exactly one context must be provided
    EXPECT_EQ(contexts.size(), 1);
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}

TEST(OpenCLUtility, get_opencl_target_version) {
    const std::regex reg{ "[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::opencl::detail::get_opencl_target_version(), reg));
}

TEST(OpenCLUtility, get_driver_version) {
    // create a valid command queue
    const std::vector<plssvm::opencl::detail::context> contexts{ plssvm::opencl::detail::get_contexts(plssvm::target_platform::automatic).first };
    const plssvm::opencl::detail::command_queue queue{ contexts[0], contexts[0].device };
    // the device name should not be empty
    const std::string driver_version = plssvm::opencl::detail::get_driver_version(queue);
    EXPECT_FALSE(driver_version.empty());
}

TEST(OpenCLUtility, get_device_name) {
    // create a valid command queue
    const std::vector<plssvm::opencl::detail::context> contexts{ plssvm::opencl::detail::get_contexts(plssvm::target_platform::automatic).first };
    const plssvm::opencl::detail::command_queue queue{ contexts[0], contexts[0].device };
    // the device name should not be empty
    const std::string name = plssvm::opencl::detail::get_device_name(queue);
    EXPECT_FALSE(name.empty());
}

TEST(OpenCLUtility, kernel_type_to_function_names) {
    // retrieve the function names
    const auto function_name_map = plssvm::opencl::detail::kernel_type_to_function_names();
    // the map must not be empty!
    EXPECT_FALSE(function_name_map.empty());
}

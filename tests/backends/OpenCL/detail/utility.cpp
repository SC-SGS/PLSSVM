/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the OpenCL backend.
 */

#include "plssvm/backends/OpenCL/detail/utility.hpp"  // plssvm::opencl::detail::{device_assert, get_contexts, get_device_name}

#include "plssvm/backends/OpenCL/exceptions.hpp"  // plssvm::opencl::backend_exception

#include "CL/cl.h"  // CL_SUCCESS, CL_DEVICE_NOT_FOUND

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_NO_THROW, EXPECT_FALSE

TEST(OpenCLUtility, device_assert) {
    // CL_SUCCESS must not throw
    EXPECT_NO_THROW(PLSSVM_OPENCL_ERROR_CHECK(CL_SUCCESS));
    EXPECT_NO_THROW(PLSSVM_OPENCL_ERROR_CHECK(CL_SUCCESS, "success!"));
    EXPECT_NO_THROW(plssvm::opencl::detail::device_assert(CL_SUCCESS));
    EXPECT_NO_THROW(plssvm::opencl::detail::device_assert(CL_SUCCESS, "success!"));

    // any other code must throw
    EXPECT_THROW_WHAT(PLSSVM_OPENCL_ERROR_CHECK(CL_DEVICE_NOT_FOUND),
                      plssvm::opencl::backend_exception,
                      "OpenCL assert 'CL_DEVICE_NOT_FOUND' (-1)!");
    EXPECT_THROW_WHAT(PLSSVM_OPENCL_ERROR_CHECK(CL_DEVICE_NOT_FOUND, "error"),
                      plssvm::opencl::backend_exception,
                      "OpenCL assert 'CL_DEVICE_NOT_FOUND' (-1): error!");
    EXPECT_THROW_WHAT(plssvm::opencl::detail::device_assert(CL_DEVICE_NOT_FOUND),
                      plssvm::opencl::backend_exception,
                      "OpenCL assert 'CL_DEVICE_NOT_FOUND' (-1)!");
    EXPECT_THROW_WHAT(plssvm::opencl::detail::device_assert(CL_DEVICE_NOT_FOUND, "error"),
                      plssvm::opencl::backend_exception,
                      "OpenCL assert 'CL_DEVICE_NOT_FOUND' (-1): error!");
}

TEST(OpenCLUtility, get_contexts) {
    const auto& [contexts, actual_target] = plssvm::opencl::detail::get_contexts(plssvm::target_platform::automatic);
    // exactly one context must be provided
    EXPECT_EQ(contexts.size(), 1);
    // the returned target must not be the automatic one
    EXPECT_NE(actual_target, plssvm::target_platform::automatic);
}

TEST(OpenCLUtility, get_device_name) {
    // create a valid command queue
    std::vector<plssvm::opencl::detail::context> contexts{ plssvm::opencl::detail::get_contexts(plssvm::target_platform::automatic).first };
    plssvm::opencl::detail::command_queue queue{ contexts[0], contexts[0].devices[0] };
    // the device name should not be empty
    const std::string name = plssvm::opencl::detail::get_device_name(queue);
    EXPECT_FALSE(name.empty());
}
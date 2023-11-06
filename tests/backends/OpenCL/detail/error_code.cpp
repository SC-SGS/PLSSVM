/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom error code implementation necessary for the OpenCL backend.
 */

#include "plssvm/backends/OpenCL/detail/error_code.hpp"  // plssvm::opencl::detail::error_code

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING

#include "CL/cl.h"        // CL_SUCCESS, CL_DEVICE_NOT_FOUND, cl_int
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_FALSE

#include <string>  // std::string

TEST(OpenCLErrorCode, default_construct) {
    // the default error code should signal success
    EXPECT_EQ(plssvm::opencl::detail::error_code{}.value(), CL_SUCCESS);
}
TEST(OpenCLErrorCode, construct) {
    // construct from an OpenCL error code
    EXPECT_EQ(plssvm::opencl::detail::error_code{ CL_DEVICE_NOT_FOUND }.value(), CL_DEVICE_NOT_FOUND);
}
TEST(OpenCLErrorCode, operator_assign) {
    // default construct the error code
    plssvm::opencl::detail::error_code errc{};
    EXPECT_EQ(errc.value(), CL_SUCCESS);
    // assign a new error code
    errc = CL_DEVICE_NOT_FOUND;
    EXPECT_EQ(errc.value(), CL_DEVICE_NOT_FOUND);
}
TEST(OpenCLErrorCode, assign) {
    // default construct the error code
    plssvm::opencl::detail::error_code errc{};
    EXPECT_EQ(errc.value(), CL_SUCCESS);
    // assign a new error code
    errc.assign(CL_DEVICE_NOT_FOUND);
    EXPECT_EQ(errc.value(), CL_DEVICE_NOT_FOUND);
}
TEST(OpenCLErrorCode, clear) {
    // construct from an OpenCL error code
    plssvm::opencl::detail::error_code errc{ CL_DEVICE_NOT_FOUND };
    EXPECT_EQ(errc.value(), CL_DEVICE_NOT_FOUND);
    // clear the error code
    errc.clear();
    EXPECT_EQ(errc.value(), CL_SUCCESS);
}

TEST(OpenCLErrorCode, value) {
    // default construct the error code
    const plssvm::opencl::detail::error_code errc{};
    // get the value
    EXPECT_EQ(errc.value(), CL_SUCCESS);
}
TEST(OpenCLErrorCode, message) {
    // default construct the error code
    const plssvm::opencl::detail::error_code errc{};
    // get the value as a string
    EXPECT_EQ(errc.message(), "CL_SUCCESS");
}
TEST(OpenCLErrorCode, possible_message_error_codes) {
    // any error code greater than zero is unknown
    std::string msg = std::string{ plssvm::opencl::detail::error_code{ 1 }.message() };
    EXPECT_EQ(msg, "UNKNOWN ERROR_CODE");
    // any error code less than -72 is unknown
    msg = std::string{ plssvm::opencl::detail::error_code{ -73 }.message() };
    EXPECT_EQ(msg, "UNKNOWN ERROR_CODE");
    // all other error codes should be available
    for (cl_int err = 0; err >= -19; --err) {
        msg = std::string{ plssvm::opencl::detail::error_code{ err }.message() };
        EXPECT_NE(msg, "UNKNOWN ERROR_CODE") << fmt::format("for OpenCL error code: {}", err);
    }
    for (cl_int err = -30; err >= -72; --err) {
        msg = std::string{ plssvm::opencl::detail::error_code{ err }.message() };
        EXPECT_NE(msg, "UNKNOWN ERROR_CODE") << fmt::format("for OpenCL error code: {}", err);
    }
}
TEST(OpenCLErrorCode, operator_bool) {
    // conversion to bool must be true if the error code is CL_SUCCESS
    EXPECT_TRUE(static_cast<bool>(plssvm::opencl::detail::error_code{}));
    // must return false otherwise
    EXPECT_FALSE(static_cast<bool>(plssvm::opencl::detail::error_code{ CL_DEVICE_NOT_FOUND }));
}

TEST(OpenCLErrorCode, operator_ostream) {
    EXPECT_CONVERSION_TO_STRING(plssvm::opencl::detail::error_code{ CL_SUCCESS }, "0: CL_SUCCESS");
    EXPECT_CONVERSION_TO_STRING(plssvm::opencl::detail::error_code{ CL_DEVICE_NOT_FOUND }, "-1: CL_DEVICE_NOT_FOUND");
}
TEST(OpenCLErrorCode, operator_equal) {
    // test two error codes for equality
    const plssvm::opencl::detail::error_code errc1{};
    const plssvm::opencl::detail::error_code errc2{ CL_SUCCESS };
    const plssvm::opencl::detail::error_code errc3{ CL_DEVICE_NOT_FOUND };

    EXPECT_TRUE(errc1 == errc1);
    EXPECT_TRUE(errc1 == errc2);
    EXPECT_FALSE(errc1 == errc3);
    EXPECT_FALSE(errc2 == errc3);
}
TEST(OpenCLErrorCode, operator_unequal) {
    // test two error codes for equality
    const plssvm::opencl::detail::error_code errc1{};
    const plssvm::opencl::detail::error_code errc2{ CL_SUCCESS };
    const plssvm::opencl::detail::error_code errc3{ CL_DEVICE_NOT_FOUND };

    EXPECT_FALSE(errc1 != errc1);
    EXPECT_FALSE(errc1 != errc2);
    EXPECT_TRUE(errc1 != errc3);
    EXPECT_TRUE(errc2 != errc3);
}
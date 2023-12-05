/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom exception classes related to the SYCL backends.
 */

#include "plssvm/backends/SYCL/exceptions.hpp"  // plssvm::sycl::backend_exception, plssvm::adaptivecpp::backend_exception, plssvm::dpcpp::backend_exception

#include "backends/generic_exceptions_tests.hpp"  // generic exception tests to instantiate

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <string_view>  // std::string_view

struct sycl_exception_test_type {
    using exception_type = plssvm::sycl::backend_exception;
    static constexpr std::string_view name = "sycl::backend_exception";
};

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(SYCLExceptions, Exception, sycl_exception_test_type);

struct adaptivecpp_exception_test_type {
    using exception_type = plssvm::adaptivecpp::backend_exception;
    static constexpr std::string_view name = "adaptivecpp::backend_exception";
};

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppExceptions, Exception, adaptivecpp_exception_test_type);

struct dpcpp_exception_test_type {
    using exception_type = plssvm::dpcpp::backend_exception;
    static constexpr std::string_view name = "dpcpp::backend_exception";
};

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPExceptions, Exception, dpcpp_exception_test_type);
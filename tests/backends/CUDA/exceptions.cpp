/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom exception classes related to the CUDA backend.
 */

#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception

#include "../generic_exceptions_tests.hpp"  // generic exception tests to instantiate

#include "gtest/gtest.h"  // INSTANTIATE_TYPED_TEST_SUITE_P

#include <string_view>  // std::string_view

struct exception_test_type {
    using exception_type = plssvm::cuda::backend_exception;
    static constexpr std::string_view name = "cuda::backend_exception";
};

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(CUDAExceptions, Exception, exception_test_type);
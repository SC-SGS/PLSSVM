/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the HIP backend.
 */

#include "backends/HIP/mock_hip_csvm.hpp"

#include "plssvm/backends/HIP/csvm.hpp"        // plssvm::hip::csvm
#include "plssvm/backends/HIP/exceptions.hpp"  // plssvm::hip::backend_exception
#include "plssvm/kernel_function_types.hpp"    // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                // plssvm::parameter
#include "plssvm/target_platforms.hpp"         // plssvm::target_platform

#include "backends/generic_tests.hpp"  // CSVM, CSVMDeathTest
#include "custom_test_macros.hpp"      // EXPECT_THROW_WHAT
#include "utility.hpp"                 // util::redirect_output

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, EXPECT_NO_THROW

template <typename T, plssvm::kernel_function_type kernel>
struct csvm_test_type {
    using mock_csvm_type = mock_hip_csvm;
    using csvm_type = plssvm::hip::csvm;
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
};

using csvm_test_types = ::testing::Types<
    csvm_test_type<float, plssvm::kernel_function_type::linear>,
    csvm_test_type<float, plssvm::kernel_function_type::polynomial>,
    csvm_test_type<float, plssvm::kernel_function_type::rbf>,
    csvm_test_type<double, plssvm::kernel_function_type::linear>,
    csvm_test_type<double, plssvm::kernel_function_type::polynomial>,
    csvm_test_type<double, plssvm::kernel_function_type::rbf>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPBackend, CSVM, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPBackendDeathTest, CSVMDeathTest, csvm_test_types);
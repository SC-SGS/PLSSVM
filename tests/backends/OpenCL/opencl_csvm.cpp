/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "backends/OpenCL/mock_opencl_csvm.hpp"

#include "backends/generic_csvm_tests.hpp"  // generic::write_model_test, generic::generate_q_test, generic::device_kernel_test, generic::predict_test, generic::accuracy_test
#include "utility.hpp"                      // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name

#include "plssvm/backends/OpenCL/csvm.hpp"   // plssvm::opencl::csvm
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST

// TODO: Konstruktors!

template <typename T, plssvm::kernel_function_type kernel>
struct csvm_test_type {
    using mock_csvm_type = mock_opencl_csvm;
    using csvm_type = plssvm::opencl::csvm;
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
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLBackend, GenericCSVM, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLBackendDeathTest, GenericCSVMDeathTest, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLBackend, GenericGPUCSVM, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLBackendDeathTest, GenericGPUCSVMDeathTest, csvm_test_types);

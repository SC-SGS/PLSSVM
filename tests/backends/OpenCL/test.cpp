/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "backends/OpenCL/mock_opencl_csvm.hpp"

#include "backends/generic_csvm_tests.hpp"  // generic::write_model_test, generic::generate_q_test, generic::device_kernel_test, generic::predict_test, generic::accuracy_test
#include "utility.hpp"                 // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name

#include "plssvm/backends/OpenCL/csvm.hpp"  // plssvm::opencl::csvm
#include "plssvm/kernel_types.hpp"          // plssvm::kernel_type
#include "plssvm/parameter.hpp"             // plssvm::parameter

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class OpenCL_CSVM : public ::testing::Test {};
TYPED_TEST_SUITE(OpenCL_CSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether the csvm factory function correctly creates an opencl::csvm
TYPED_TEST(OpenCL_CSVM, csvm_factory) {
    generic::csvm_factory_test<plssvm::opencl::csvm, typename TypeParam::real_type, plssvm::backend_type::opencl>();
}

// check whether writing the resulting model file is correct
TYPED_TEST(OpenCL_CSVM, write_model) {
    generic::write_model_test<mock_opencl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the q vector is generated correctly
TYPED_TEST(OpenCL_CSVM, generate_q) {
    generic::generate_q_test<mock_opencl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the device kernels are correct
TYPED_TEST(OpenCL_CSVM, device_kernel) {
    generic::device_kernel_test<mock_opencl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the correct labels are predicted
TYPED_TEST(OpenCL_CSVM, predict) {
    generic::predict_test<mock_opencl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the accuracy calculation is correct
TYPED_TEST(OpenCL_CSVM, accuracy) {
    generic::accuracy_test<mock_opencl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

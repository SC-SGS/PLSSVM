/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the SYCL backend.
 */

#include "backends/SYCL/mock_sycl_csvm.hpp"

#include "backends/generic_tests.hpp"  // generic::write_model_test, generic::generate_q_test, generic::device_kernel_test, generic::predict_test, generic::accuracy_test
#include "utility.hpp"                 // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name

#include "plssvm/backends/SYCL/csvm.hpp"                    // plssvm::sycl::csvm
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter

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
class SYCL_CSVM : public ::testing::Test {};
TYPED_TEST_SUITE(SYCL_CSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether the std::string <-> plssvm::sycl::kernel_invocation_type conversions are correct
TEST(SYCL_CSVM, kernel_invocation_type) {
    // check conversions to std::string
    util::gtest_expect_enum_to_string_string_conversion(plssvm::sycl::kernel_invocation_type::automatic, "automatic");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::sycl::kernel_invocation_type::nd_range, "nd_range");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::sycl::kernel_invocation_type::hierarchical, "hierarchical");
    util::gtest_expect_enum_to_string_string_conversion(static_cast<plssvm::sycl::kernel_invocation_type>(3), "unknown");

    // check conversion from std::string
    util::gtest_expect_string_to_enum_conversion("automatic", plssvm::sycl::kernel_invocation_type::automatic);
    util::gtest_expect_string_to_enum_conversion("AUTOMATIC", plssvm::sycl::kernel_invocation_type::automatic);
    util::gtest_expect_string_to_enum_conversion("nd_range", plssvm::sycl::kernel_invocation_type::nd_range);
    util::gtest_expect_string_to_enum_conversion("ND_RANGE", plssvm::sycl::kernel_invocation_type::nd_range);
    util::gtest_expect_string_to_enum_conversion("hierarchical", plssvm::sycl::kernel_invocation_type::hierarchical);
    util::gtest_expect_string_to_enum_conversion("HIERARCHICAL", plssvm::sycl::kernel_invocation_type::hierarchical);
    util::gtest_expect_string_to_enum_conversion<plssvm::sycl::kernel_invocation_type>("foo");
}

// check whether the csvm factory function correctly creates a sycl::csvm
TYPED_TEST(SYCL_CSVM, csvm_factory) {
    generic::csvm_factory_test<plssvm::sycl::csvm, typename TypeParam::real_type, plssvm::backend_type::sycl>();
}

// check whether writing the resulting model file is correct
TYPED_TEST(SYCL_CSVM, write_model) {
    generic::write_model_test<mock_sycl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the q vector is generated correctly
TYPED_TEST(SYCL_CSVM, generate_q) {
    generic::generate_q_test<mock_sycl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the nd_range device kernels are correct
TYPED_TEST(SYCL_CSVM, device_kernel_nd_range) {
    generic::device_kernel_test<mock_sycl_csvm, typename TypeParam::real_type, TypeParam::kernel, plssvm::sycl::kernel_invocation_type::nd_range>();
}
// check whether the hierarchical device kernels are correct
TYPED_TEST(SYCL_CSVM, device_kernel_hierarchical) {
    generic::device_kernel_test<mock_sycl_csvm, typename TypeParam::real_type, TypeParam::kernel, plssvm::sycl::kernel_invocation_type::hierarchical>();
}

// check whether the correct labels are predicted
TYPED_TEST(SYCL_CSVM, predict) {
    generic::predict_test<mock_sycl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the accuracy calculation is correct
TYPED_TEST(SYCL_CSVM, accuracy) {
    generic::accuracy_test<mock_sycl_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}
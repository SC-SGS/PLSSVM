/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Tests for the generic functionality related to the SYCL backend.
*/



#include "backends/generic_csvm_tests.hpp"  // generic::write_model_test, generic::generate_q_test, generic::device_kernel_test, generic::predict_test, generic::accuracy_test
#include "utility.hpp"                      // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name

#include "plssvm/backends/SYCL/exceptions.hpp"              // plssvm::sycl::backend_exception
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl_generic::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST

#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    #include "backends/SYCL/DPCPP/mock_dpcpp_csvm.hpp"  // mock_dpcpp_csvm
    #include "plssvm/backends/SYCL/DPCPP/csvm.hpp"        // plssvm::dpcpp::csvm

template <typename T, plssvm::kernel_function_type kernel>
struct dpcpp_csvm_test_type {
    using mock_csvm_type = mock_dpcpp_csvm;
    using csvm_type = plssvm::dpcpp::csvm;
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
};

using dpcpp_csvm_test_types = ::testing::Types<
    dpcpp_csvm_test_type<float, plssvm::kernel_function_type::linear>,
    dpcpp_csvm_test_type<float, plssvm::kernel_function_type::polynomial>,
    dpcpp_csvm_test_type<float, plssvm::kernel_function_type::rbf>,
    dpcpp_csvm_test_type<double, plssvm::kernel_function_type::linear>,
    dpcpp_csvm_test_type<double, plssvm::kernel_function_type::polynomial>,
    dpcpp_csvm_test_type<double, plssvm::kernel_function_type::rbf>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericCSVM, dpcpp_csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericCSVMDeathTest, dpcpp_csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericGPUCSVM, dpcpp_csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericGPUCSVMDeathTest, dpcpp_csvm_test_types);
#endif

#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    #include "backends/SYCL/hipSYCL/mock_hipsycl_csvm.hpp"  // mock_hipsycl_csvm
    #include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"        // plssvm::hipsycl::csvm

template <typename T, plssvm::kernel_function_type kernel>
struct hipsycl_csvm_test_type {
    using mock_csvm_type = mock_hipsycl_csvm;
    using csvm_type = plssvm::hipsycl::csvm;
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
};

using hipsycl_csvm_test_types = ::testing::Types<
    hipsycl_csvm_test_type<float, plssvm::kernel_function_type::linear>,
    hipsycl_csvm_test_type<float, plssvm::kernel_function_type::polynomial>,
    hipsycl_csvm_test_type<float, plssvm::kernel_function_type::rbf>,
    hipsycl_csvm_test_type<double, plssvm::kernel_function_type::linear>,
    hipsycl_csvm_test_type<double, plssvm::kernel_function_type::polynomial>,
    hipsycl_csvm_test_type<double, plssvm::kernel_function_type::rbf>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLCSVM, GenericCSVM, hipsycl_csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLCSVMDeathTest, GenericCSVMDeathTest, hipsycl_csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLCSVM, GenericGPUCSVM, hipsycl_csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLCSVMDeathTest, GenericGPUCSVMDeathTest, hipsycl_csvm_test_types);
#endif
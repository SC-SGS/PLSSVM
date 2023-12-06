/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the SYCL backend using AdaptiveCpp as SYCL implementation.
 */

#include "backends/SYCL/AdaptiveCpp/mock_adaptivecpp_csvm.hpp"

#include "plssvm/backend_types.hpp"                         // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/SYCL/AdaptiveCpp/csvm.hpp"        // plssvm::adaptivecpp::csvm
#include "plssvm/backends/SYCL/exceptions.hpp"              // plssvm::adaptivecpp::backend_exception
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "backends/generic_csvm_tests.hpp"      // generic CSVM tests to instantiate
#include "backends/generic_gpu_csvm_tests.hpp"  // generic GPU CSVM tests to instantiate
#include "custom_test_macros.hpp"               // EXPECT_THROW_WHAT
#include "naming.hpp"                           // naming::test_parameter_to_name
#include "types_to_test.hpp"                    // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "utility.hpp"                          // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

class AdaptiveCppCSVM : public ::testing::Test, private util::redirect_output<> {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(AdaptiveCppCSVM, construct_parameter) {
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::adaptivecpp::csvm{ plssvm::parameter{} });
}
TEST_F(AdaptiveCppCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

    // every target is allowed for SYCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu, params }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia, params }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd, params }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel, params }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(AdaptiveCppCSVM, construct_target_and_named_args) {
    // every target is allowed for SYCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::cpu,
                                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                                  plssvm::cost = 2.0,
                                                  plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_nvidia,
                                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                                  plssvm::cost = 2.0,
                                                  plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_amd,
                                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                                  plssvm::cost = 2.0,
                                                  plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::adaptivecpp::csvm{ plssvm::target_platform::gpu_intel,
                                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                                  plssvm::cost = 2.0,
                                                  plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::adaptivecpp::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TEST_F(AdaptiveCppCSVM, get_kernel_invocation_type) {
    // construct default CSVM
    const plssvm::adaptivecpp::csvm svm{ plssvm::parameter{} };

    // after construction: get_kernel_invocation_type must refer to a plssvm::sycl::kernel_invocation_type that is not automatic
    EXPECT_NE(svm.get_kernel_invocation_type(), plssvm::sycl::kernel_invocation_type::automatic);
}

struct adaptivecpp_csvm_test_type {
    using mock_csvm_type = mock_adaptivecpp_csvm;
    using csvm_type = plssvm::adaptivecpp::csvm;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline static constexpr auto additional_arguments = std::make_tuple();
};
using adaptivecpp_csvm_test_tuple = std::tuple<adaptivecpp_csvm_test_type>;
using adaptivecpp_csvm_test_label_type_list = util::cartesian_type_product_t<adaptivecpp_csvm_test_tuple, plssvm::detail::supported_label_types>;
using adaptivecpp_csvm_test_type_list = util::cartesian_type_product_t<adaptivecpp_csvm_test_tuple>;

// the tests used in the instantiated GTest test suites
using adaptivecpp_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<adaptivecpp_csvm_test_type_list>;
using adaptivecpp_solver_type_gtest = util::combine_test_parameters_gtest_t<adaptivecpp_csvm_test_type_list, util::solver_type_list>;
using adaptivecpp_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<adaptivecpp_csvm_test_type_list, util::kernel_function_type_list>;
using adaptivecpp_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<adaptivecpp_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
using adaptivecpp_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<adaptivecpp_csvm_test_label_type_list, util::kernel_function_and_classification_type_list>;
using adaptivecpp_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<adaptivecpp_csvm_test_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;

// instantiate type-parameterized tests
// generic CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericCSVM, adaptivecpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericCSVMSolver, adaptivecpp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericCSVMKernelFunction, adaptivecpp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericCSVMSolverKernelFunction, adaptivecpp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericCSVMKernelFunctionClassification, adaptivecpp_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericCSVMSolverKernelFunctionClassification, adaptivecpp_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVMDeathTest, GenericCSVMSolverDeathTest, adaptivecpp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, adaptivecpp_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericGPUCSVM, adaptivecpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(AdaptiveCppCSVM, GenericGPUCSVMKernelFunction, adaptivecpp_kernel_function_type_gtest, naming::test_parameter_to_name);

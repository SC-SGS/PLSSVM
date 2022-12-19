/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the SYCL backend using hipSYCL as SYCL implementation.
 */

#include "backends/SYCL/hipSYCL/mock_hipsycl_csvm.hpp"

#include "plssvm/backends/SYCL/exceptions.hpp"              // plssvm::hipsycl::backend_exception
#include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"            // plssvm::hipsycl::csvm
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "backends/generic_csvm_tests.hpp"  // generic CSVM tests to instantiate
#include "custom_test_macros.hpp"           // EXPECT_THROW_WHAT
#include "utility.hpp"                      // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, TYPED_TEST_SUITE, TYPED_TEST, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::{Test, Types}

// TODO: constructor tests

// class @PLSSVM_SYCL_BACKEND_INCLUDE_NAME@CSVM : public ::testing::Test, private util::redirect_output<> {};
//
//// check whether the constructor correctly fails when using an incompatible target platform
// TEST_F(@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@CSVM, construct_parameter) {
//     // the automatic target platform must always be available
//     EXPECT_NO_THROW(plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::parameter{} });
// }
// TEST_F(@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@CSVM, construct_target_and_parameter) {
//     // create parameter struct
//     const plssvm::parameter params{};
//
//     // every target is allowed for SYCL
// #if defined(PLSSVM_HAS_CPU_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::cpu, params }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::cpu, params }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// #if defined(PLSSVM_HAS_NVIDIA_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_nvidia, params }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_nvidia, params }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// #if defined(PLSSVM_HAS_AMD_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_amd, params }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_amd, params }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// #if defined(PLSSVM_HAS_INTEL_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_intel, params }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_intel, params }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// }
// TEST_F(@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@CSVM, construct_target_and_named_args) {
//     // every target is allowed for SYCL
// #if defined(PLSSVM_HAS_CPU_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// #if defined(PLSSVM_HAS_NVIDIA_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// #if defined(PLSSVM_HAS_AMD_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::cost = 2.0 }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// #if defined(PLSSVM_HAS_INTEL_TARGET)
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
//     EXPECT_NO_THROW((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }));
// #else
//     EXPECT_THROW_WHAT((plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
//                       plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::backend_exception,
//                       "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
// #endif
// }

template <typename T, plssvm::kernel_function_type kernel>
struct csvm_test_type {
    using mock_csvm_type = mock_hipsycl_csvm;
    using csvm_type = plssvm::hipsycl::csvm;
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
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLBackend, GenericCSVM, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLBackendDeathTest, GenericCSVMDeathTest, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLBackend, GenericGPUCSVM, csvm_test_types);
INSTANTIATE_TYPED_TEST_SUITE_P(hipSYCLBackendDeathTest, GenericGPUCSVMDeathTest, csvm_test_types);

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "backends/OpenCL/mock_opencl_csvm.hpp"

#include "plssvm/backend_types.hpp"                // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/OpenCL/csvm.hpp"         // plssvm::opencl::csvm
#include "plssvm/backends/OpenCL/exceptions.hpp"   // plssvm::opencl::backend_exception
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/type_list.hpp"             // plssvm::detail::label_type_list
#include "plssvm/kernel_function_types.hpp"        // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"             // plssvm::target_platform

#include "backends/generic_csvm_tests.hpp"      // generic CSVM tests to instantiate
#include "backends/generic_gpu_csvm_tests.hpp"  // generic GPU CSVM tests to instantiate
#include "custom_test_macros.hpp"               // EXPECT_THROW_WHAT
#include "naming.hpp"                           // naming::test_parameter_to_name
#include "types_to_test.hpp"                    // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "utility.hpp"                          // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

class OpenCLCSVM : public ::testing::Test, private util::redirect_output<> {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(OpenCLCSVM, construct_parameter) {
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::opencl::csvm{ plssvm::parameter{} });
}
TEST_F(OpenCLCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

    // every target is allowed for OpenCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::cpu, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_nvidia, params }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_amd, params }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_intel, params }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(OpenCLCSVM, construct_target_and_named_args) {
    // every target is allowed for OpenCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::opencl::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::opencl::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

struct opencl_csvm_test_type {
    using mock_csvm_type = mock_opencl_csvm;
    using csvm_type = plssvm::opencl::csvm;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline static constexpr auto additional_arguments = std::make_tuple();
};
using opencl_csvm_test_tuple = std::tuple<opencl_csvm_test_type>;
using opencl_csvm_test_label_type_list = util::cartesian_type_product_t<opencl_csvm_test_tuple, plssvm::detail::supported_label_types>;
using opencl_csvm_test_type_list = util::cartesian_type_product_t<opencl_csvm_test_tuple>;

// the tests used in the instantiated GTest test suites
using opencl_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list>;
using opencl_solver_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list, util::solver_type_list>;
using opencl_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list, util::kernel_function_type_list>;
using opencl_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
using opencl_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_label_type_list, util::kernel_function_and_classification_type_list>;
using opencl_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;

// instantiate type-parameterized tests
// generic CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVM, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMSolver, opencl_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMKernelFunction, opencl_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMSolverKernelFunction, opencl_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMKernelFunctionClassification, opencl_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMSolverKernelFunctionClassification, opencl_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericCSVMSolverDeathTest, opencl_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, opencl_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericGPUCSVM, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericGPUCSVMKernelFunction, opencl_kernel_function_type_gtest, naming::test_parameter_to_name);

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

#include "plssvm/backend_types.hpp"                // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/HIP/csvm.hpp"            // plssvm::hip::csvm
#include "plssvm/backends/HIP/exceptions.hpp"      // plssvm::hip::backend_exception
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
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

class HIPCSVM : public ::testing::Test, private util::redirect_output<> {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(HIPCSVM, construct_parameter) {
#if defined(PLSSVM_HAS_AMD_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::hip::csvm{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT(plssvm::hip::csvm{ plssvm::parameter{} },
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(HIPCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_AMD_TARGET)
    // only automatic or gpu_amd are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((plssvm::hip::csvm{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((plssvm::hip::csvm{ plssvm::target_platform::gpu_amd, params }));
#else
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::automatic, params }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::cpu, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'cpu' for the HIP backend!");
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the HIP backend!");
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HIP backend!");
}
TEST_F(HIPCSVM, construct_target_and_named_args) {
#if defined(PLSSVM_HAS_AMD_TARGET)
    // only automatic or gpu_amd are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((plssvm::hip::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::hip::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'cpu' for the HIP backend!");
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the HIP backend!");
    EXPECT_THROW_WHAT((plssvm::hip::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HIP backend!");
}

struct hip_csvm_test_type {
    using mock_csvm_type = mock_hip_csvm;
    using csvm_type = plssvm::hip::csvm;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline static constexpr auto additional_arguments = std::make_tuple();
};
using hip_csvm_test_tuple = std::tuple<hip_csvm_test_type>;
using hip_csvm_test_label_type_list = util::cartesian_type_product_t<hip_csvm_test_tuple, plssvm::detail::supported_label_types>;
using hip_csvm_test_type_list = util::cartesian_type_product_t<hip_csvm_test_tuple>;

// the tests used in the instantiated GTest test suites
using hip_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list>;
using hip_solver_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list, util::solver_type_list>;
using hip_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list, util::kernel_function_type_list>;
using hip_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
using hip_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_label_type_list, util::kernel_function_and_classification_type_list>;
using hip_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;

// instantiate type-parameterized tests
// generic CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVM, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMSolver, hip_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMKernelFunction, hip_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMSolverKernelFunction, hip_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMKernelFunctionClassification, hip_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMSolverKernelFunctionClassification, hip_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericCSVMSolverDeathTest, hip_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, hip_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericGPUCSVM, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericGPUCSVMKernelFunction, hip_kernel_function_type_gtest, naming::test_parameter_to_name);

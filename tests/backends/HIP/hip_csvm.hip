/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the HIP backend.
 */

#include "plssvm/backends/HIP/csvm.hpp"            // plssvm::hip::{csvm, csvc, csvr}
#include "plssvm/backends/HIP/exceptions.hpp"      // plssvm::hip::backend_exception
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"        // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"             // plssvm::target_platform

#include "tests/backends/generic_base_csvc_tests.hpp"  // generic C-SVC tests to instantiate
#include "tests/backends/generic_base_csvm_tests.hpp"  // generic C-SVM tests to instantiate
#include "tests/backends/generic_base_csvr_tests.hpp"  // generic C-SVR tests to instantiate
#include "tests/backends/generic_gpu_csvm_tests.hpp"   // generic GPU C-SVM tests to instantiate
#include "tests/backends/HIP/mock_hip_csvm.hpp"        // mock_hip_csvm
#include "tests/custom_test_macros.hpp"                // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                           // util::redirect_output

#include "gtest/gtest.h"  // TYPED_TEST, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

using hip_csvm_types_list = std::tuple<plssvm::hip::csvc, plssvm::hip::csvr>;
using hip_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<hip_csvm_types_list>>;

template <typename T>
class HIPCSVMConstructor : public ::testing::Test,
                           private util::redirect_output<> {
  protected:
    using fixture_csvm_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(HIPCSVMConstructor, hip_csvm_types_gtest, naming::test_parameter_to_name);

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(HIPCSVMConstructor, construct_parameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HIP_BACKEND_USE_HIP_RUNTIME)
    #if defined(PLSSVM_HAS_AMD_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
    #else
    EXPECT_THROW_WHAT(csvm_type{ plssvm::parameter{} },
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif
#elif defined(PLSSVM_HIP_BACKEND_USE_CUDA_RUNTIME)
    #if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
    #else
    EXPECT_THROW_WHAT(csvm_type{ plssvm::parameter{} },
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif
#endif
}

TYPED_TEST(HIPCSVMConstructor, construct_target_and_parameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HIP_BACKEND_USE_HIP_RUNTIME)
    #if defined(PLSSVM_HAS_AMD_TARGET)
    // only automatic or gpu_amd are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, params }));
    #else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, params }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'cpu' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HIP backend using the HIP runtime!");
#elif defined(PLSSVM_HIP_BACKEND_USE_CUDA_RUNTIME)
    #if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, params }));
    #else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, params }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'cpu' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_amd' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HIP backend using the HIP runtime!");
#endif
}

TYPED_TEST(HIPCSVMConstructor, construct_named_args) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HIP_BACKEND_USE_HIP_RUNTIME)
    #if defined(PLSSVM_HAS_AMD_TARGET)
    // only automatic or gpu_amd are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::cost = 2.0 }));
    #else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif
#elif defined(PLSSVM_HIP_BACKEND_USE_CUDA_RUNTIME)
    #if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::cost = 2.0 }));
    #else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif
#endif
}

TYPED_TEST(HIPCSVMConstructor, construct_target_and_named_args) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HIP_BACKEND_USE_HIP_RUNTIME)
    #if defined(PLSSVM_HAS_AMD_TARGET)
    // only automatic or gpu_amd are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }));
    #else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'cpu' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HIP backend using the HIP runtime!");
#elif defined(PLSSVM_HIP_BACKEND_USE_CUDA_RUNTIME)
    #if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the HIP backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
    #else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    #endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'cpu' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_amd' for the HIP backend using the HIP runtime!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::hip::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HIP backend using the HIP runtime!");
#endif
}

template <bool mock_grid_size>
struct hip_csvm_test_type {
    using mock_csvm_type = mock_hip_csvm<mock_grid_size>;
    using csvm_type = plssvm::hip::csvm;
    using csvc_type = plssvm::hip::csvc;
    using csvr_type = plssvm::hip::csvr;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline constexpr static auto additional_arguments = std::make_tuple();
};

// a tuple containing the test structs
using hip_csvm_test_tuple = std::tuple<hip_csvm_test_type<false>>;

// the tests used in the instantiated GTest test suites
// general test types
using hip_csvm_test_type_list = util::cartesian_type_product_t<hip_csvm_test_tuple>;
using hip_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list>;
using hip_solver_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list, util::solver_type_list>;
using hip_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list, util::kernel_function_type_list>;
using hip_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using hip_csvm_test_classification_label_type_list = util::cartesian_type_product_t<hip_csvm_test_tuple, util::classification_label_types>;
using hip_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using hip_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using hip_csvm_test_regression_label_type_list = util::cartesian_type_product_t<hip_csvm_test_tuple, util::regression_label_types>;
using hip_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using hip_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVM, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMKernelFunction, hip_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMSolver, hip_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericCSVMSolverKernelFunction, hip_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVC, GenericCSVC, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVC, GenericCSVCKernelFunctionClassification, hip_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVC, GenericCSVCSolverKernelFunctionClassification, hip_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVR, GenericCSVR, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVR, GenericCSVRKernelFunction, hip_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVR, GenericCSVRSolverKernelFunction, hip_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericCSVMDeathTest, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericCSVMSolverDeathTest, hip_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, hip_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, hip_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU C-SVM tests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericGPUCSVM, hip_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVM, GenericGPUCSVMKernelFunction, hip_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU C-SVM DeathTests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMDeathTest, GenericGPUCSVMDeathTest, hip_csvm_test_type_gtest, naming::test_parameter_to_name);

using hip_mock_csvm_test_tuple = std::tuple<hip_csvm_test_type<true>>;
using hip_mock_csvm_test_type_list = util::cartesian_type_product_t<hip_mock_csvm_test_tuple>;

using hip_mock_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<hip_mock_csvm_test_type_list>;
using hip_mock_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hip_mock_csvm_test_type_list, util::kernel_function_type_list>;

// generic GPU C-SVM tests - mocked grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMFakedGridSize, GenericGPUCSVM, hip_mock_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HIPCSVMFakedGridSize, GenericGPUCSVMKernelFunction, hip_mock_kernel_function_type_gtest, naming::test_parameter_to_name);

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"               // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/OpenCL/csvm.hpp"        // plssvm::opencl::{csvm, csvc, csvr}
#include "plssvm/backends/OpenCL/exceptions.hpp"  // plssvm::opencl::backend_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "tests/backends/generic_base_csvc_tests.hpp"  // generic C-SVC tests to instantiate
#include "tests/backends/generic_base_csvm_tests.hpp"  // generic C-SVM tests to instantiate
#include "tests/backends/generic_base_csvr_tests.hpp"  // generic C-SVR tests to instantiate
#include "tests/backends/generic_gpu_csvm_tests.hpp"   // generic GPU C-SVM tests to instantiate
#include "tests/backends/OpenCL/mock_opencl_csvm.hpp"  // mock_opencl_csvm
#include "tests/custom_test_macros.hpp"                // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                           // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

using opencl_csvm_types_list = std::tuple<plssvm::opencl::csvc, plssvm::opencl::csvr>;
using opencl_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<opencl_csvm_types_list>>;

template <typename T>
class OpenCLCSVMConstructor : public ::testing::Test,
                              private util::redirect_output<> {
  protected:
    using fixture_csvm_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(OpenCLCSVMConstructor, opencl_csvm_types_gtest, naming::test_parameter_to_name);

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(OpenCLCSVMConstructor, DefaultConstruct) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // default constructor must always work
    EXPECT_NO_THROW(csvm_type{});
}

TYPED_TEST(OpenCLCSVMConstructor, ConstructParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
}

TYPED_TEST(OpenCLCSVMConstructor, ConstructTargetAndParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // create parameter struct
    const plssvm::parameter params{};

    // every target is allowed for OpenCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(OpenCLCSVMConstructor, ConstructNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // every target is allowed for OpenCL
    EXPECT_NO_THROW((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::cost = 2.0 }));
}

TYPED_TEST(OpenCLCSVMConstructor, ConstructTargetAndNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // every target is allowed for OpenCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::opencl::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

template <bool mock_grid_size>
struct opencl_csvm_test_type {
    using mock_csvm_type = mock_opencl_csvm<mock_grid_size>;
    using csvm_type = plssvm::opencl::csvm;
    using csvc_type = plssvm::opencl::csvc;
    using csvr_type = plssvm::opencl::csvr;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    constexpr static auto additional_arguments = std::make_tuple();
};

// a tuple containing the test structs
using opencl_csvm_test_tuple = std::tuple<opencl_csvm_test_type<false>>;

// the tests used in the instantiated GTest test suites
// general test types
using opencl_csvm_test_type_list = util::cartesian_type_product_t<opencl_csvm_test_tuple>;
using opencl_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list>;
using opencl_solver_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list, util::solver_type_list>;
using opencl_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list, util::kernel_function_type_list>;
using opencl_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using opencl_csvm_test_classification_label_type_list = util::cartesian_type_product_t<opencl_csvm_test_tuple, util::classification_label_types>;
using opencl_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using opencl_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using opencl_csvm_test_regression_label_type_list = util::cartesian_type_product_t<opencl_csvm_test_tuple, util::regression_label_types>;
using opencl_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using opencl_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVM, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMKernelFunction, opencl_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMSolver, opencl_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericCSVMSolverKernelFunction, opencl_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVC, GenericCSVC, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVC, GenericCSVCKernelFunctionClassification, opencl_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVC, GenericCSVCSolverKernelFunctionClassification, opencl_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVR, GenericCSVR, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVR, GenericCSVRKernelFunction, opencl_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVR, GenericCSVRSolverKernelFunction, opencl_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericCSVMDeathTest, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericCSVMSolverDeathTest, opencl_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, opencl_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, opencl_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU C-SVM tests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericGPUCSVM, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVM, GenericGPUCSVMKernelFunction, opencl_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU C-SVM DeathTests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMDeathTest, GenericGPUCSVMDeathTest, opencl_csvm_test_type_gtest, naming::test_parameter_to_name);

using opencl_mock_csvm_test_tuple = std::tuple<opencl_csvm_test_type<true>>;
using opencl_mock_csvm_test_type_list = util::cartesian_type_product_t<opencl_mock_csvm_test_tuple>;

using opencl_mock_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<opencl_mock_csvm_test_type_list>;
using opencl_mock_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<opencl_mock_csvm_test_type_list, util::kernel_function_type_list>;

// generic GPU C-SVM tests - mocked grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMFakedGridSize, GenericGPUCSVM, opencl_mock_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenCLCSVMFakedGridSize, GenericGPUCSVMKernelFunction, opencl_mock_kernel_function_type_gtest, naming::test_parameter_to_name);

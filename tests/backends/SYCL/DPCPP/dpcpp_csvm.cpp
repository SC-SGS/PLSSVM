/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the SYCL backend using DPC++ as SYCL implementation.
 */

#include "plssvm/backend_types.hpp"                        // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"             // plssvm::dpcpp::{csvm, csvc, csvr}
#include "plssvm/backends/SYCL/exceptions.hpp"             // plssvm::dpcpp::backend_exception
#include "plssvm/detail/arithmetic_type_name.hpp"          // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"                // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                            // plssvm::parameter, plssvm::kernel_type, plssvm::cost, plssvm::sycl_data_parallel_kernel
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform

#include "tests/backends/generic_base_csvc_tests.hpp"     // generic C-SVC tests to instantiate
#include "tests/backends/generic_base_csvm_tests.hpp"     // generic C-SVM tests to instantiate
#include "tests/backends/generic_base_csvr_tests.hpp"     // generic C-SVR tests to instantiate
#include "tests/backends/generic_gpu_csvm_tests.hpp"      // generic GPU C-SVM tests to instantiate
#include "tests/backends/SYCL/DPCPP/mock_dpcpp_csvm.hpp"  // mock_dpcpp_csvm
#include "tests/custom_test_macros.hpp"                   // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                               // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                        // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                              // util::redirect_output

#include "gtest/gtest.h"  // TYPED_TEST, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

using dpcpp_csvm_types_list = std::tuple<plssvm::dpcpp::csvc, plssvm::dpcpp::csvr>;
using dpcpp_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<dpcpp_csvm_types_list>>;

template <typename T>
class DPCPPCSVMConstructor : public ::testing::Test,
                             private util::redirect_output<> {
  protected:
    using fixture_csvm_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(DPCPPCSVMConstructor, dpcpp_csvm_types_gtest, naming::test_parameter_to_name);

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(DPCPPCSVMConstructor, DefaultConstruct) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // default constructor must always work
    EXPECT_NO_THROW(csvm_type{});
    EXPECT_NO_THROW((csvm_type{ plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
}

TYPED_TEST(DPCPPCSVMConstructor, ConstructParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
    EXPECT_NO_THROW((csvm_type{ plssvm::parameter{}, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
}

TYPED_TEST(DPCPPCSVMConstructor, ConstructTargetAndParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // create parameter struct
    const plssvm::parameter params{};

    // every target is allowed for SYCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(DPCPPCSVMConstructor, ConstructNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // every target is allowed for SYCL
    EXPECT_NO_THROW((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
}

TYPED_TEST(DPCPPCSVMConstructor, ConstructTargetAndNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // every target is allowed for SYCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu,
                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                  plssvm::cost = 2.0,
                                  plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia,
                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                  plssvm::cost = 2.0,
                                  plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd,
                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                  plssvm::cost = 2.0,
                                  plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel,
                                  plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                  plssvm::cost = 2.0,
                                  plssvm::sycl_data_parallel_kernel = plssvm::sycl::data_parallel_kernel::work_group }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(DPCPPCSVMConstructor, GetDataParallelKernel) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // construct default C-SVM
    const csvm_type svm{ plssvm::parameter{} };

    // after construction: get_data_parallel_kernel must refer to a plssvm::sycl::data_parallel_kernel that is not automatic
    EXPECT_NE(svm.get_data_parallel_kernel(), plssvm::sycl::data_parallel_kernel::automatic);
}

template <bool mock_grid_size, plssvm::sycl::data_parallel_kernel data_parallel_kernel_type>
struct dpcpp_csvm_test_type {
    using mock_csvm_type = mock_dpcpp_csvm<mock_grid_size>;
    using csvm_type = plssvm::dpcpp::csvm;
    using csvc_type = plssvm::dpcpp::csvc;
    using csvr_type = plssvm::dpcpp::csvr;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline static auto additional_arguments = std::make_tuple(std::make_pair(plssvm::sycl_data_parallel_kernel, data_parallel_kernel_type));
};

// a tuple containing the test structs
using dpcpp_csvm_test_tuple = std::tuple<
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
    dpcpp_csvm_test_type<false, plssvm::sycl::data_parallel_kernel::hierarchical>,
#endif
    dpcpp_csvm_test_type<false, plssvm::sycl::data_parallel_kernel::basic>,
    dpcpp_csvm_test_type<false, plssvm::sycl::data_parallel_kernel::work_group>>;

// the tests used in the instantiated GTest test suites
// general test types
using dpcpp_csvm_test_type_list = util::cartesian_type_product_t<dpcpp_csvm_test_tuple>;
using dpcpp_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_type_list>;
using dpcpp_solver_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_type_list, util::solver_type_list>;
using dpcpp_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_type_list, util::kernel_function_type_list>;
using dpcpp_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using dpcpp_csvm_test_classification_label_type_list = util::cartesian_type_product_t<dpcpp_csvm_test_tuple, util::classification_label_types>;
using dpcpp_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using dpcpp_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using dpcpp_csvm_test_regression_label_type_list = util::cartesian_type_product_t<dpcpp_csvm_test_tuple, util::regression_label_types>;
using dpcpp_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using dpcpp_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericCSVM, dpcpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericCSVMKernelFunction, dpcpp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericCSVMSolver, dpcpp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericCSVMSolverKernelFunction, dpcpp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVC, GenericCSVC, dpcpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVC, GenericCSVCKernelFunctionClassification, dpcpp_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVC, GenericCSVCSolverKernelFunctionClassification, dpcpp_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVR, GenericCSVR, dpcpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVR, GenericCSVRKernelFunction, dpcpp_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVR, GenericCSVRSolverKernelFunction, dpcpp_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericCSVMDeathTest, dpcpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericCSVMSolverDeathTest, dpcpp_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, dpcpp_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, dpcpp_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU C-SVM tests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericGPUCSVM, dpcpp_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVM, GenericGPUCSVMKernelFunction, dpcpp_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU C-SVM DeathTests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMDeathTest, GenericGPUCSVMDeathTest, dpcpp_csvm_test_type_gtest, naming::test_parameter_to_name);

using dpcpp_mock_csvm_test_tuple = std::tuple<
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
    dpcpp_csvm_test_type<true, plssvm::sycl::data_parallel_kernel::hierarchical>,
#endif
    dpcpp_csvm_test_type<true, plssvm::sycl::data_parallel_kernel::basic>,
    dpcpp_csvm_test_type<true, plssvm::sycl::data_parallel_kernel::work_group>>;

using dpcpp_mock_csvm_test_type_list = util::cartesian_type_product_t<dpcpp_mock_csvm_test_tuple>;

using dpcpp_mock_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_mock_csvm_test_type_list>;
using dpcpp_mock_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<dpcpp_mock_csvm_test_type_list, util::kernel_function_type_list>;

// generic GPU C-SVM tests - mocked grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMFakedGridSize, GenericGPUCSVM, dpcpp_mock_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPCSVMFakedGridSize, GenericGPUCSVMKernelFunction, dpcpp_mock_kernel_function_type_gtest, naming::test_parameter_to_name);

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the HPX backend.
 */

#include "plssvm/backend_types.hpp"                                                // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/HPX/csvm.hpp"                                            // plssvm::hpx::{csvm, csvc, csvr}
#include "plssvm/backends/HPX/exceptions.hpp"                                      // plssvm::hpx::backend_exception
#include "plssvm/backends/HPX/kernel/cg_explicit/blas.hpp"                         // plssvm::hpx::{device_kernel_symm, device_ce_kernel_symm_mirror}
#include "plssvm/backends/HPX/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::hpx::device_kernel_assembly
#include "plssvm/backends/HPX/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::hpx::device_kernel_assembly_symm
#include "plssvm/backends/HPX/kernel/predict_kernel.hpp"                           // plssvm::hpx::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                    // plssvm::PADDING_SIZE
#include "plssvm/kernel_function_types.hpp"                                        // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                                       // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                                                    // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"                                             // plssvm::target_platform

#include "tests/backends/generic_base_csvc_tests.hpp"  // generic C-SVC tests to instantiate
#include "tests/backends/generic_base_csvm_tests.hpp"  // generic C-SVM tests to instantiate
#include "tests/backends/generic_base_csvr_tests.hpp"  // generic C-SVR tests to instantiate
#include "tests/backends/HPX/mock_hpx_csvm.hpp"        // mock_hpx_csvm
#include "tests/custom_test_macros.hpp"                // EXPECT_THROW_WHAT
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"                           // util::redirect_output

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

using hpx_csvm_types_list = std::tuple<plssvm::hpx::csvc, plssvm::hpx::csvr>;
using hpx_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<hpx_csvm_types_list>>;

template <typename T>
class HPXCSVMConstructor : public ::testing::Test,
                           private util::redirect_output<> {
  protected:
    using fixture_csvm_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(HPXCSVMConstructor, hpx_csvm_types_gtest, naming::test_parameter_to_name);

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(HPXCSVMConstructor, DefaultConstruct) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    // default constructor must always work
    EXPECT_NO_THROW(csvm_type{});
#else
    EXPECT_THROW_WHAT((csvm_type{}),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(HPXCSVMConstructor, ConstructParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::parameter{} }),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(HPXCSVMConstructor, ConstructTargetAndParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the HPX backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, params }),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params }),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::hpx::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the HPX backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::hpx::backend_exception,
                      "Invalid target platform 'gpu_amd' for the HPX backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::hpx::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HPX backend!");
}

TYPED_TEST(HPXCSVMConstructor, ConstructNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the HPX backend
    EXPECT_NO_THROW((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(HPXCSVMConstructor, ConstructTargetAndNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the HPX backend
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::hpx::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::hpx::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the HPX backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::hpx::backend_exception,
                      "Invalid target platform 'gpu_amd' for the HPX backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::hpx::backend_exception,
                      "Invalid target platform 'gpu_intel' for the HPX backend!");
}

struct hpx_csvm_test_type {
    using mock_csvm_type = mock_hpx_csvm;
    using csvm_type = plssvm::hpx::csvm;
    using csvc_type = plssvm::hpx::csvc;
    using csvr_type = plssvm::hpx::csvr;
    using device_ptr_type = const plssvm::soa_matrix<plssvm::real_type> *;
    inline constexpr static auto additional_arguments = std::make_tuple();
};

// a tuple containing the test structs
using hpx_csvm_test_tuple = std::tuple<hpx_csvm_test_type>;

// the tests used in the instantiated GTest test suites
// general test types
using hpx_csvm_test_type_list = util::cartesian_type_product_t<hpx_csvm_test_tuple>;
using hpx_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_type_list>;
using hpx_solver_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_type_list, util::solver_type_list>;
using hpx_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_type_list, util::kernel_function_type_list>;
using hpx_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using hpx_csvm_test_classification_label_type_list = util::cartesian_type_product_t<hpx_csvm_test_tuple, util::classification_label_types>;
using hpx_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using hpx_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using hpx_csvm_test_regression_label_type_list = util::cartesian_type_product_t<hpx_csvm_test_tuple, util::regression_label_types>;
using hpx_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using hpx_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<hpx_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVM, GenericCSVM, hpx_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVM, GenericCSVMKernelFunction, hpx_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVM, GenericCSVMSolver, hpx_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVM, GenericCSVMSolverKernelFunction, hpx_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVC, GenericCSVC, hpx_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVC, GenericCSVCKernelFunctionClassification, hpx_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVC, GenericCSVCSolverKernelFunctionClassification, hpx_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVR, GenericCSVR, hpx_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVR, GenericCSVRKernelFunction, hpx_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVR, GenericCSVRSolverKernelFunction, hpx_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVMDeathTest, GenericCSVMDeathTest, hpx_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVMDeathTest, GenericCSVMSolverDeathTest, hpx_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, hpx_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, hpx_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// define the exact functions to be used in the generic header
using plssvm::hpx::detail::device_kernel_assembly;
using plssvm::hpx::detail::device_kernel_assembly_symm;
using plssvm::hpx::detail::device_kernel_predict;
using plssvm::hpx::detail::device_kernel_predict_linear;
using plssvm::hpx::detail::device_kernel_symm;
using plssvm::hpx::detail::device_kernel_symm_mirror;
using plssvm::hpx::detail::device_kernel_w_linear;
#include "tests/backends/generic_csvm_tests.hpp"  // generic backend C-SVM tests to instantiate

// generic non-GPU C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVM, GenericBackendCSVM, hpx_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVM, GenericBackendCSVMKernelFunction, hpx_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic non-GPU C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVMDeathTest, GenericBackendCSVMDeathTest, hpx_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(HPXCSVMDeathTest, GenericBackendCSVMKernelFunctionDeathTest, hpx_kernel_function_type_gtest, naming::test_parameter_to_name);

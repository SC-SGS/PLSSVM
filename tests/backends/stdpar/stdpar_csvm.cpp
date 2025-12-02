/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the stdpar backend.
 */

#include "plssvm/backend_types.hpp"                                                   // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/stdpar/csvm.hpp"                                            // plssvm::stdpar::{csvm, csvc, csvr}
#include "plssvm/backends/stdpar/kernel/cg_explicit/blas.hpp"                         // plssvm::stdpar::{device_kernel_symm, device_ce_kernel_symm_mirror}
#include "plssvm/backends/stdpar/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::stdpar::device_kernel_assembly
#include "plssvm/backends/stdpar/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::stdpar::device_kernel_assembly_symm
#include "plssvm/backends/stdpar/kernel/predict_kernel.hpp"                           // plssvm::stdpar::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                       // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                                          // plssvm::soa_matrix. plssvm::aos_matrix
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform

#include "tests/backends/generic_base_csvc_tests.hpp"  // generic C-SVC tests to instantiate
#include "tests/backends/generic_base_csvm_tests.hpp"  // generic C-SVM tests to instantiate
#include "tests/backends/generic_base_csvr_tests.hpp"  // generic C-SVR tests to instantiate
#include "tests/backends/stdpar/mock_stdpar_csvm.hpp"  // mock_stdpar_csvm
#include "tests/naming.hpp"                            // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                     // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <cstddef>  // std::size_t
#include <tuple>    // std::make_tuple, std::tuple
#include <vector>   // std::vector

struct stdpar_csvm_test_type {
    using mock_csvm_type = mock_stdpar_csvm;
    using csvm_type = plssvm::stdpar::csvm;
    using csvc_type = plssvm::stdpar::csvc;
    using csvr_type = plssvm::stdpar::csvr;
    using device_ptr_type = const plssvm::soa_matrix<plssvm::real_type> *;
    inline constexpr static auto additional_arguments = std::make_tuple();
};

// a tuple containing the test structs
using stdpar_csvm_test_tuple = std::tuple<stdpar_csvm_test_type>;

// the tests used in the instantiated GTest test suites
// general test types
using stdpar_csvm_test_type_list = util::cartesian_type_product_t<stdpar_csvm_test_tuple>;
using stdpar_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list>;
using stdpar_solver_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list, util::solver_type_list>;
using stdpar_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list, util::kernel_function_type_list>;
using stdpar_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
// C-SVC specific test types
using stdpar_csvm_test_classification_label_type_list = util::cartesian_type_product_t<stdpar_csvm_test_tuple, util::classification_label_types>;
using stdpar_classification_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_classification_label_type_list, util::kernel_function_and_classification_type_list>;
using stdpar_classification_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_classification_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;
// C-SVR specific test types
using stdpar_csvm_test_regression_label_type_list = util::cartesian_type_product_t<stdpar_csvm_test_tuple, util::regression_label_types>;
using stdpar_regression_label_type_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_regression_label_type_list, util::kernel_function_type_list>;
using stdpar_regression_label_type_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<stdpar_csvm_test_regression_label_type_list, util::solver_and_kernel_function_type_list>;

// instantiate type-parameterized tests
// generic C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVM, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVMKernelFunction, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVMSolver, stdpar_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericCSVMSolverKernelFunction, stdpar_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
// generic C-SVC tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVC, GenericCSVC, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVC, GenericCSVCKernelFunctionClassification, stdpar_classification_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVC, GenericCSVCSolverKernelFunctionClassification, stdpar_classification_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
// generic C-SVR tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVR, GenericCSVR, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVR, GenericCSVRKernelFunction, stdpar_regression_label_type_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVR, GenericCSVRSolverKernelFunction, stdpar_regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMDeathTest, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMSolverDeathTest, stdpar_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, stdpar_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// TODO: better without that much code cuplication
// define the exact functions to be used in the generic header
template <plssvm::kernel_function_type kernel_function, typename... Args>
void device_kernel_assembly(plssvm::real_type *kernel_matrix, const plssvm::soa_matrix<plssvm::real_type> &data, const std::size_t device_num_rows, const std::size_t device_row_offset, const std::vector<plssvm::real_type> &q, const plssvm::real_type QA_cost, const plssvm::real_type cost, Args... kernel_function_parameter) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_assembly<plssvm::target_platform::gpu_nvidia, kernel_function, Args...>{}(kernel_matrix, data, device_num_rows, device_row_offset, q, QA_cost, cost, kernel_function_parameter...);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_assembly<plssvm::target_platform::gpu_amd, kernel_function, Args...>{}(kernel_matrix, data, device_num_rows, device_row_offset, q, QA_cost, cost, kernel_function_parameter...);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_assembly<plssvm::target_platform::gpu_intel, kernel_function, Args...>{}(kernel_matrix, data, device_num_rows, device_row_offset, q, QA_cost, cost, kernel_function_parameter...);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_assembly<plssvm::target_platform::cpu, kernel_function, Args...>{}(kernel_matrix, data, device_num_rows, device_row_offset, q, QA_cost, cost, kernel_function_parameter...);
            break;
    }
}

void device_kernel_symm(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t device_num_rows, const std::size_t device_row_offset, const plssvm::real_type alpha, const plssvm::real_type *A, const plssvm::soa_matrix<plssvm::real_type> &B, const plssvm::real_type beta, plssvm::soa_matrix<plssvm::real_type> &C) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_symm<plssvm::target_platform::gpu_nvidia>{}(num_rows, num_rhs, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_symm<plssvm::target_platform::gpu_amd>{}(num_rows, num_rhs, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_symm<plssvm::target_platform::gpu_intel>{}(num_rows, num_rhs, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_symm<plssvm::target_platform::cpu>{}(num_rows, num_rhs, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
    }
}

void device_kernel_symm_mirror(const std::size_t num_rows, const std::size_t num_rhs, const std::size_t num_mirror_rows, const std::size_t device_num_rows, const std::size_t device_row_offset, const plssvm::real_type alpha, const plssvm::real_type *A, const plssvm::soa_matrix<plssvm::real_type> &B, const plssvm::real_type beta, plssvm::soa_matrix<plssvm::real_type> &C) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_symm_mirror<plssvm::target_platform::gpu_nvidia>{}(num_rows, num_rhs, num_mirror_rows, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_symm_mirror<plssvm::target_platform::gpu_amd>{}(num_rows, num_rhs, num_mirror_rows, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_symm_mirror<plssvm::target_platform::gpu_intel>{}(num_rows, num_rhs, num_mirror_rows, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_symm_mirror<plssvm::target_platform::cpu>{}(num_rows, num_rhs, num_mirror_rows, device_num_rows, device_row_offset, alpha, A, B, beta, C);
            break;
    }
}

template <plssvm::kernel_function_type kernel_function, typename... Args>
void device_kernel_assembly_symm(const plssvm::real_type alpha, const std::vector<plssvm::real_type> &q, const plssvm::soa_matrix<plssvm::real_type> &data, const std::size_t device_num_rows, const std::size_t device_row_offset, const plssvm::real_type QA_cost, const plssvm::real_type cost, const plssvm::soa_matrix<plssvm::real_type> &B, plssvm::soa_matrix<plssvm::real_type> &C, Args... kernel_function_parameter) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_assembly_symm<plssvm::target_platform::gpu_nvidia, kernel_function, Args...>{}(alpha, q, data, device_num_rows, device_row_offset, QA_cost, cost, B, C, kernel_function_parameter...);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_assembly_symm<plssvm::target_platform::gpu_amd, kernel_function, Args...>{}(alpha, q, data, device_num_rows, device_row_offset, QA_cost, cost, B, C, kernel_function_parameter...);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_assembly_symm<plssvm::target_platform::gpu_intel, kernel_function, Args...>{}(alpha, q, data, device_num_rows, device_row_offset, QA_cost, cost, B, C, kernel_function_parameter...);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_assembly_symm<plssvm::target_platform::cpu, kernel_function, Args...>{}(alpha, q, data, device_num_rows, device_row_offset, QA_cost, cost, B, C, kernel_function_parameter...);
            break;
    }
}

void device_kernel_w_linear(plssvm::soa_matrix<plssvm::real_type> &w, const plssvm::aos_matrix<plssvm::real_type> &alpha, const plssvm::soa_matrix<plssvm::real_type> &support_vectors, const std::size_t device_num_sv, const std::size_t device_sv_offset) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_w_linear<plssvm::target_platform::gpu_nvidia>{}(w, alpha, support_vectors, device_num_sv, device_sv_offset);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_w_linear<plssvm::target_platform::gpu_amd>{}(w, alpha, support_vectors, device_num_sv, device_sv_offset);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_w_linear<plssvm::target_platform::gpu_intel>{}(w, alpha, support_vectors, device_num_sv, device_sv_offset);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_w_linear<plssvm::target_platform::cpu>{}(w, alpha, support_vectors, device_num_sv, device_sv_offset);
            break;
    }
}

void device_kernel_predict_linear(plssvm::aos_matrix<plssvm::real_type> &prediction, const plssvm::soa_matrix<plssvm::real_type> &w, const std::vector<plssvm::real_type> &rho, const plssvm::soa_matrix<plssvm::real_type> &predict_points, const std::size_t device_num_predict_points, const std::size_t device_row_offset) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_predict_linear<plssvm::target_platform::gpu_nvidia>{}(prediction, w, rho, predict_points, device_num_predict_points, device_row_offset);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_predict_linear<plssvm::target_platform::gpu_amd>{}(prediction, w, rho, predict_points, device_num_predict_points, device_row_offset);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_predict_linear<plssvm::target_platform::gpu_intel>{}(prediction, w, rho, predict_points, device_num_predict_points, device_row_offset);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_predict_linear<plssvm::target_platform::cpu>{}(prediction, w, rho, predict_points, device_num_predict_points, device_row_offset);
            break;
    }
}

template <plssvm::kernel_function_type kernel_function, typename... Args>
void device_kernel_predict(plssvm::aos_matrix<plssvm::real_type> &prediction, const plssvm::aos_matrix<plssvm::real_type> &alpha, const std::vector<plssvm::real_type> &rho, const plssvm::soa_matrix<plssvm::real_type> &support_vectors, const plssvm::soa_matrix<plssvm::real_type> &predict_points, const std::size_t device_num_predict_points, const std::size_t device_row_offset, Args... kernel_function_parameter) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            // may never be reached
            break;
        case plssvm::target_platform::gpu_nvidia:
            plssvm::stdpar::detail::device_kernel_predict<plssvm::target_platform::gpu_nvidia, kernel_function, Args...>{}(prediction, alpha, rho, support_vectors, predict_points, device_num_predict_points, device_row_offset, kernel_function_parameter...);
            break;
        case plssvm::target_platform::gpu_amd:
            plssvm::stdpar::detail::device_kernel_predict<plssvm::target_platform::gpu_amd, kernel_function, Args...>{}(prediction, alpha, rho, support_vectors, predict_points, device_num_predict_points, device_row_offset, kernel_function_parameter...);
            break;
        case plssvm::target_platform::gpu_intel:
            plssvm::stdpar::detail::device_kernel_predict<plssvm::target_platform::gpu_intel, kernel_function, Args...>{}(prediction, alpha, rho, support_vectors, predict_points, device_num_predict_points, device_row_offset, kernel_function_parameter...);
            break;
        case plssvm::target_platform::cpu:
            plssvm::stdpar::detail::device_kernel_predict<plssvm::target_platform::cpu, kernel_function, Args...>{}(prediction, alpha, rho, support_vectors, predict_points, device_num_predict_points, device_row_offset, kernel_function_parameter...);
            break;
    }
}

#include "tests/backends/generic_csvm_tests.hpp"  // generic backend C-SVM tests to instantiate

// generic non-GPU C-SVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericBackendCSVM, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVM, GenericBackendCSVMKernelFunction, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic non-GPU C-SVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericBackendCSVMDeathTest, stdpar_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(stdparCSVMDeathTest, GenericBackendCSVMKernelFunctionDeathTest, stdpar_kernel_function_type_gtest, naming::test_parameter_to_name);

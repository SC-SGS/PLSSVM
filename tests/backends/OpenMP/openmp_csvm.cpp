/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "backends/OpenMP/mock_openmp_csvm.hpp"

#include "plssvm/backends/OpenMP/csvm.hpp"        // plssvm::openmp::csvm
#include "plssvm/backends/OpenMP/exceptions.hpp"  // plssvm::openmp::backend_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "naming.hpp"
#include "types_to_test.hpp"
#include "backends/generic_tests.hpp"  // generic::write_model_test, generic::generate_q_test, generic::predict_test, generic::accuracy_test
#include "custom_test_macros.hpp"
#include "mock_csvm.hpp"  // mock_csvm
#include "utility.hpp"    // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name, util::gtest_assert_floating_point_near, EXPECT_THROW_WHAT

#include "fmt/core.h"
#include "fmt/ostream.h"
#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ

#include <random>  // std::random_device, std::mt19937, std::uniform_real_distribution
#include <vector>  // std::vector

template <typename T, plssvm::kernel_function_type kernel>
struct parameter_definition {
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
};

class parameter_definition_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}__{}", plssvm::detail::arithmetic_type_name<typename T::real_type>(), T::kernel_type);
    }
};

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    parameter_definition<float, plssvm::kernel_function_type::linear>,
    parameter_definition<float, plssvm::kernel_function_type::polynomial>,
    parameter_definition<float, plssvm::kernel_function_type::rbf>,
    parameter_definition<double, plssvm::kernel_function_type::linear>,
    parameter_definition<double, plssvm::kernel_function_type::polynomial>,
    parameter_definition<double, plssvm::kernel_function_type::rbf>>;

// TODO: kernel??!??!?

class OpenMPCSVM : public ::testing::Test, private util::redirect_output {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(OpenMPCSVM, construct_parameter_invalid_target_platform) {
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW(mock_openmp_csvm{ plssvm::target_platform::automatic });
    EXPECT_NO_THROW(mock_openmp_csvm{ plssvm::target_platform::cpu });

    // all other target platforms must throw
    EXPECT_THROW_WHAT(mock_openmp_csvm{ plssvm::target_platform::gpu_nvidia },
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT(mock_openmp_csvm{ plssvm::target_platform::gpu_amd },
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT(mock_openmp_csvm{ plssvm::target_platform::gpu_intel },
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

// TODO: test constructors?!

template <typename T>
class OpenMPCSVMSolveSystemOfLinearEquations : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMSolveSystemOfLinearEquations, parameter_types, parameter_definition_to_name);

TYPED_TEST(OpenMPCSVMSolveSystemOfLinearEquations, solve_system_of_linear_equations_diagonal) {
    generic::test_solve_system_of_linear_equations<typename TypeParam::real_type, plssvm::openmp::csvm>(TypeParam::kernel_type);
}

template <typename T>
class OpenMPCSVMGenerateQ : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMGenerateQ, parameter_types, parameter_definition_to_name);

TYPED_TEST(OpenMPCSVMGenerateQ, generate_q) {
    using real_type = typename TypeParam::real_type;
    const plssvm::kernel_function_type kernel_type = TypeParam::kernel_type;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel_type, 2, 0.001, 1.0, 0.1 };

    // create the data that should be used
    const plssvm::data_set<real_type> data{ PLSSVM_TEST_FILE };

    // calculate correct q vector (ground truth)
    const std::vector<real_type> ground_truth = compare::generate_q(params, data.data());

    // create C-SVM: must be done using the mock class, since plssvm::openmp::csvm::generate_q is protected
    const mock_openmp_csvm svm;

    // calculate the q vector using the OpenMP backend
    const std::vector<real_type> calculated = svm.generate_q(params, data.data());

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
}

template <typename T>
class OpenMPCSVMRunDeviceKernel : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMRunDeviceKernel, parameter_types, parameter_definition_to_name);

TYPED_TEST(OpenMPCSVMRunDeviceKernel, run_device_kernel) {
    using real_type = typename TypeParam::real_type;
    const plssvm::kernel_function_type kernel_type = TypeParam::kernel_type;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel_type, 2, 0.001, 1.0, 0.1 };

    // create the data that should be used
    const plssvm::data_set<real_type> data{ PLSSVM_TEST_FILE };
    const std::vector<real_type> rhs = util::generate_random_vector<real_type>(data.num_data_points() - 1, real_type{ 1.0 }, real_type{ 2.0 });
    const std::vector<real_type> q = compare::generate_q(params, data.data());
    const real_type QA_cost = compare::kernel_function(params, data.data().back(), data.data().back()) + 1 / params.cost;

    // create C-SVM: must be done using the mock class, since plssvm::openmp::csvm::calculate_w is protected
    const mock_openmp_csvm svm;

    for (const real_type add : { real_type{ -1.0 }, real_type{ 1.0 } }) {
        // calculate the correct device function result
        const std::vector<real_type> ground_truth = compare::device_kernel_function(params, data.data(), rhs, q, QA_cost, add);

        // perform the kernel calculation on the device
        std::vector<real_type> calculated(data.num_data_points() - 1);
        svm.run_device_kernel(params, q, calculated, rhs, data.data(), QA_cost, add);

        // check the calculated result for correctness
        EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
    }
}

template <typename T>
class OpenMPCSVMCalculateW : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMCalculateW, util::real_type_gtest , naming::real_type_to_name);

TYPED_TEST(OpenMPCSVMCalculateW, calculate_w) {
    using real_type = TypeParam;

    // create the data that should be used
    const plssvm::data_set<real_type> support_vectors{ PLSSVM_TEST_FILE };
    const std::vector<real_type> weights = util::generate_random_vector<real_type>(support_vectors.num_data_points());

    // calculate the correct w vector
    const std::vector<real_type> ground_truth = compare::calculate_w(support_vectors.data(), weights);

    // create C-SVM: must be done using the mock class, since plssvm::openmp::csvm::calculate_w is protected
    const mock_openmp_csvm svm;

    // calculate the w vector using the OpenMP backend
    const std::vector<real_type> calculated = svm.calculate_w(support_vectors.data(), weights);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
}

template <typename T>
class OpenMPCSVMPredictAndScore : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMPredictAndScore, parameter_types, parameter_definition_to_name);

TYPED_TEST(OpenMPCSVMPredictAndScore, predict) {
    generic::test_predict<typename TypeParam::real_type, plssvm::openmp::csvm>(TypeParam::kernel_type);
}

TYPED_TEST(OpenMPCSVMPredictAndScore, score) {
    generic::test_score<typename TypeParam::real_type, plssvm::openmp::csvm>(TypeParam::kernel_type);
}
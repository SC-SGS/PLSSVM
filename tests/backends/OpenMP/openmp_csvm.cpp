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
    using real_type = typename TypeParam::real_type;
    const plssvm::kernel_function_type kernel_type = TypeParam::kernel_type;

    // create parameter struct
    plssvm::detail::parameter<real_type> params{};
    params.kernel_type = kernel_type;
    params.cost = 2.0;

    // create the data that should be used
    // Matrix with 1-1/cost on main diagonal. Thus, the diagonal entries become one with the additional addition of 1/cost
    const std::vector<std::vector<real_type>> data = {
        { real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) } }
    };
    const std::vector<real_type> rhs{ real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 }, real_type{ 4.0 }, real_type{ 5.0 } };

    // create C-SVM: must be done using the mock class, since plssvm::openmp::csvm::solve_system_of_linear_equations_impl is protected
    const mock_openmp_csvm svm;

    // solve the system of linear equations using the CG algorithm: (A + (I * 1 / cost))x =b  => Ix = b => should be trivial x = b
    const std::pair<std::vector<real_type>, real_type> result = svm.solve_system_of_linear_equations_impl(params, data, rhs, real_type{ 0.00001 }, 1);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(rhs, result.first);
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

    // check the calculated result for correctness result for
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

        // check the calculated result for correctness result for
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

    // check the calculated result for correctness result for
    EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
}


//// check whether the device kernels are correct
// TYPED_TEST(OpenMPCSVM, device_kernel) {
//     // create parameter object
//     plssvm::parameter<typename TypeParam::real_type> params;
//     params.print_info = false;
//     params.kernel = TypeParam::kernel;
//
//     params.parse_train_file(PLSSVM_TEST_FILE);
//
//     // create base C-SVM
//     mock_csvm csvm{ params };
//     using real_type = typename decltype(csvm)::real_type;
//
//     const std::size_t dept = csvm.get_num_data_points() - 1;
//
//     // create x vector and fill it with random values
//     std::vector<real_type> x(dept);
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<real_type> dist(1.0, 2.0);
//     std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
//
//     // create C-SVM using the OpenMP backend
//     mock_openmp_csvm csvm_openmp{ params };
//
//     // create correct q vector, cost and QA_cost
//     const std::vector<real_type> q_vec = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm_openmp.get_num_devices(), csvm);
//     const real_type cost = csvm.get_cost();
//     const real_type QA_cost = compare::kernel_function<TypeParam::kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;
//
//     // setup data on device
//     csvm_openmp.setup_data_on_device();
//
//     for (const auto add : { real_type{ -1 }, real_type{ 1 } }) {
//         const std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);
//
//         std::vector<real_type> calculated(dept, 0.0);
//         csvm_openmp.set_QA_cost(QA_cost);
//         csvm_openmp.set_cost(cost);
//         csvm_openmp.run_device_kernel(q_vec, calculated, x, csvm_openmp.get_device_data(), add);
//
//         ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
//         for (typename std::vector<real_type>::size_type index = 0; index < correct.size(); ++index) {
//             util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
//         }
//     }
// }
//
//// check whether the correct labels are predicted
// TYPED_TEST(OpenMPCSVM, predict) {
//     generic::predict_test<mock_openmp_csvm, typename TypeParam::real_type, TypeParam::kernel>();
// }
//
//// check whether the accuracy calculation is correct
// TYPED_TEST(OpenMPCSVM, accuracy) {
//     generic::accuracy_test<mock_openmp_csvm, typename TypeParam::real_type, TypeParam::kernel>();
// }

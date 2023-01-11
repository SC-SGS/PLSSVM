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

#include "plssvm/backend_types.hpp"                // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/OpenMP/csvm.hpp"         // plssvm::openmp::csvm
#include "plssvm/backends/OpenMP/exceptions.hpp"   // plssvm::openmp::backend_exception
#include "plssvm/data_set.hpp"                     // plssvm::data_set
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"        // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"             // plssvm::target_platform

#include "../../custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "../../naming.hpp"              // naming::{real_type_kernel_function_to_name, real_type_to_name}
#include "../../types_to_test.hpp"       // util::{real_type_kernel_function_gtest, real_type_gtest}
#include "../../utility.hpp"             // util::{redirect_output, generate_random_vector}
#include "../compare.hpp"                // compare::{generate_q, calculate_w, kernel_function, device_kernel_function}
#include "../generic_csvm_tests.hpp"     // generic::{test_solve_system_of_linear_equations, test_predict_values, test_predict, test_score}

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, TYPED_TEST_SUITE, TYPED_TEST, ::testing::Test

#include <tuple>   // std::make_tuple
#include <vector>  // std::vector

class OpenMPCSVM : public ::testing::Test, private util::redirect_output<> {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(OpenMPCSVM, construct_parameter) {
#if defined(PLSSVM_HAS_CPU_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::openmp::csvm{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT(plssvm::openmp::csvm{ plssvm::parameter{} },
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(OpenMPCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT(plssvm::openmp::csvm{ plssvm::target_platform::automatic, params },
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT(plssvm::openmp::csvm{ plssvm::target_platform::cpu, params },
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}
TEST_F(OpenMPCSVM, construct_target_and_named_args) {
#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the OpenMP backend
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::openmp::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT(plssvm::openmp::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 },
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT(plssvm::openmp::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 },
                      plssvm::openmp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    EXPECT_THROW_WHAT((plssvm::openmp::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::openmp::backend_exception,
                      "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

template <typename T, plssvm::kernel_function_type kernel>
struct csvm_test_type {
    using mock_csvm_type = mock_openmp_csvm;
    using csvm_type = plssvm::openmp::csvm;
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
    inline static auto additional_arguments = std::make_tuple();
};

class csvm_test_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}_{}_{}",
                           plssvm::csvm_to_backend_type_v<typename T::csvm_type>,
                           plssvm::detail::arithmetic_type_name<typename T::real_type>(),
                           T::kernel_type);
    }
};

using csvm_test_types = ::testing::Types<
    csvm_test_type<float, plssvm::kernel_function_type::linear>,
    csvm_test_type<float, plssvm::kernel_function_type::polynomial>,
    csvm_test_type<float, plssvm::kernel_function_type::rbf>,
    csvm_test_type<double, plssvm::kernel_function_type::linear>,
    csvm_test_type<double, plssvm::kernel_function_type::polynomial>,
    csvm_test_type<double, plssvm::kernel_function_type::rbf>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPBackend, GenericCSVM, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMPBackendDeathTest, GenericCSVMDeathTest, csvm_test_types, csvm_test_type_to_name);

template <typename T>
class OpenMPCSVMGenerateQ : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMGenerateQ, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

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
    const mock_openmp_csvm svm{};

    // calculate the q vector using the OpenMP backend
    const std::vector<real_type> calculated = svm.generate_q(params, data.data());

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, ground_truth);
}

template <typename T>
class OpenMPCSVMCalculateW : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMCalculateW, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(OpenMPCSVMCalculateW, calculate_w) {
    using real_type = TypeParam;

    // create the data that should be used
    const plssvm::data_set<real_type> support_vectors{ PLSSVM_TEST_FILE };
    const std::vector<real_type> weights = util::generate_random_vector<real_type>(support_vectors.num_data_points());

    // calculate the correct w vector
    const std::vector<real_type> ground_truth = compare::calculate_w(support_vectors.data(), weights);

    // create C-SVM: must be done using the mock class, since plssvm::openmp::csvm::calculate_w is protected
    const mock_openmp_csvm svm{};

    // calculate the w vector using the OpenMP backend
    const std::vector<real_type> calculated = svm.calculate_w(support_vectors.data(), weights);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(calculated, ground_truth, real_type{ 1.0e5 });
}

template <typename T>
class OpenMPCSVMRunDeviceKernel : public OpenMPCSVM {};
TYPED_TEST_SUITE(OpenMPCSVMRunDeviceKernel, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

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
    const mock_openmp_csvm svm{};

    for (const real_type add : { real_type{ -1.0 }, real_type{ 1.0 } }) {
        // calculate the correct device function result
        const std::vector<real_type> ground_truth = compare::device_kernel_function(params, data.data(), rhs, q, QA_cost, add);

        // perform the kernel calculation on the device
        std::vector<real_type> calculated(data.num_data_points() - 1);
        svm.run_device_kernel(params, q, calculated, rhs, data.data(), QA_cost, add);

        // check the calculated result for correctness
        EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, ground_truth);
    }
}
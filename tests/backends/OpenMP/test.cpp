/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "mock_openmp_csvm.hpp"

#include "../../mock_csvm.hpp"   // mock_csvm
#include "../../utility.hpp"     // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name, util::gtest_assert_floating_point_near, EXPECT_THROW_WHAT
#include "../generic_tests.hpp"  // generic::write_model_test, generic::generate_q_test, generic::predict_test, generic::accuracy_test

#include "plssvm/backends/OpenMP/csvm.hpp"        // plssvm::openmp::csvm
#include "plssvm/backends/OpenMP/exceptions.hpp"  // plssvm::openmp::backend_exception
#include "plssvm/kernel_types.hpp"                // plssvm::kernel_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/target_platform.hpp"             // plssvm::target_platform

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ

#include <random>  // std::random_device, std::mt19937, std::uniform_real_distribution
#include <vector>  // std::vector

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class OpenMP_CSVM : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMP_CSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether the csvm factory function correctly creates an openmp::csvm
TYPED_TEST(OpenMP_CSVM, csvm_factory) {
    generic::csvm_factory_test<plssvm::openmp::csvm, typename TypeParam::real_type, plssvm::backend_type::openmp>();
}

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(OpenMP_CSVM, constructor_invalid_target_platform) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // only automatic or cpu are allowed as target platform for the OpenMP backend
    params.target = plssvm::target_platform::automatic;
    EXPECT_NO_THROW(mock_openmp_csvm{ params });
    params.target = plssvm::target_platform::cpu;
    EXPECT_NO_THROW(mock_openmp_csvm{ params });
    params.target = plssvm::target_platform::gpu_nvidia;
    EXPECT_THROW_WHAT(mock_openmp_csvm{ params }, plssvm::openmp::backend_exception, "Invalid target platform 'gpu_nvidia' for the OpenMP backend!");
    params.target = plssvm::target_platform::gpu_amd;
    EXPECT_THROW_WHAT(mock_openmp_csvm{ params }, plssvm::openmp::backend_exception, "Invalid target platform 'gpu_amd' for the OpenMP backend!");
    params.target = plssvm::target_platform::gpu_intel;
    EXPECT_THROW_WHAT(mock_openmp_csvm{ params }, plssvm::openmp::backend_exception, "Invalid target platform 'gpu_intel' for the OpenMP backend!");
}

// check whether writing the resulting model file is correct
TYPED_TEST(OpenMP_CSVM, write_model) {
    generic::write_model_test<mock_openmp_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the q vector is generated correctly
TYPED_TEST(OpenMP_CSVM, generate_q) {
    generic::generate_q_test<mock_openmp_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the device kernels are correct
TYPED_TEST(OpenMP_CSVM, device_kernel) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_FILE);

    // create base C-SVM
    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;

    const std::size_t dept = csvm.get_num_data_points() - 1;

    // create x vector and fill it with random values
    std::vector<real_type> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(1.0, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    // create correct q vector, cost and QA_cost
    const std::vector<real_type> q_vec = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = compare::kernel_function<TypeParam::kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;

    // create C-SVM using the OpenMP backend
    mock_openmp_csvm csvm_openmp{ params };

    // setup data on device
    csvm_openmp.setup_data_on_device();

    for (const auto add : { real_type{ -1 }, real_type{ 1 } }) {
        const std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        std::vector<real_type> calculated(dept, 0.0);
        csvm_openmp.set_QA_cost(QA_cost);
        csvm_openmp.set_cost(cost);
        csvm_openmp.run_device_kernel(q_vec, calculated, x, csvm_openmp.get_device_data(), add);

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (typename std::vector<real_type>::size_type index = 0; index < correct.size(); ++index) {
            util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
        }
    }
}

// check whether the correct labels are predicted
TYPED_TEST(OpenMP_CSVM, predict) {
    generic::predict_test<mock_openmp_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

// check whether the accuracy calculation is correct
TYPED_TEST(OpenMP_CSVM, accuracy) {
    generic::accuracy_test<mock_openmp_csvm, typename TypeParam::real_type, TypeParam::kernel>();
}

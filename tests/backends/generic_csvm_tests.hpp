/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_TESTS_HPP_
#pragma once

#include "plssvm/constants.hpp"              // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE;
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/detail/layout.hpp"          // plssvm::detail::{layout_type, transform_to_layout}
#include "plssvm/detail/operators.hpp"       // operators namespace
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/model.hpp"                  // plssvm::model
#include "plssvm/parameter.hpp"              // plssvm::cost, plssvm::kernel_type, plssvm::parameter, plssvm::detail::parameter

#include "../custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR, EXPECT_FLOATING_POINT_VECTOR_EQ
#include "../utility.hpp"             // util::{redirect_output, generate_random_vector, construct_from_tuple}
#include "compare.hpp"                // compare::{generate_q, calculate_w, kernel_function, device_kernel_function}

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // ASSERT_EQ, EXPECT_EQ, EXPECT_TRUE, TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P,
                          // ::testing::Test

#include <cmath>        // std::sqrt, std::abs
#include <cstddef>      // std::size_t
#include <fstream>      // std::ifstream
#include <iterator>     // std::istream_iterator
#include <limits>       // std::numeric_limits::epsilon
#include <tuple>        // std::ignore
#include <vector>       // std::vector

//*************************************************************************************************************************************//
//                                                                 CSVM                                                                //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVM : public ::testing::Test, protected util::redirect_output<> {};
TYPED_TEST_SUITE_P(GenericCSVM);

TYPED_TEST_P(GenericCSVM, solve_system_of_linear_equations_trivial) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create parameter struct
    plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1.0;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        GTEST_SKIP() << "solve_system_of_linear_equations_trivial currently doesn't work with the rbf kernel!";
    }

    // create the data that should be used
    // Matrix with 1-1/cost on main diagonal. Thus, the diagonal entries become one with the additional addition of 1/cost
    const std::vector<std::vector<real_type>> A = {
        { real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) } },
    };
    const std::vector<real_type> rhs{ real_type{ 1.0 }, real_type{ -1.0 }, real_type{ 1.0 }, real_type{ -1.0 } };

    // create C-SVM: must be done using the mock class, since solve_system_of_linear_equations_impl is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    // solve the system of linear equations using the CG algorithm:
    // | Q  1 |  *  | a |  =  | y |
    // | 1  0 |     | b |     | 0 |
    // with Q = A^TA
    const auto &[calculated_x, calculated_rho] = svm.solve_system_of_linear_equations(params, A, rhs, real_type{ 0.00001 }, A.front().size());

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated_x, rhs);
    // EXPECT_FLOATING_POINT_NEAR(calculated_rho, real_type{ 0.0 });
    EXPECT_FLOATING_POINT_NEAR(std::abs(calculated_rho) - std::numeric_limits<real_type>::epsilon(), std::numeric_limits<real_type>::epsilon());
}

TYPED_TEST_P(GenericCSVM, solve_system_of_linear_equations) {
    GTEST_SKIP() << "currently not implemented";
    // TODO: add non-trivial test
}

TYPED_TEST_P(GenericCSVM, solve_system_of_linear_equations_with_correction) {
    GTEST_SKIP() << "currently not implemented";
    // TODO: test for the correction scheme after 50 iterations
}

TYPED_TEST_P(GenericCSVM, predict_values) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create parameter struct
    plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1.0;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        GTEST_SKIP() << "predict_values currently doesn't work with the rbf kernel!";
    }

    // create the data that should be used
    const std::vector<std::vector<real_type>> support_vectors = {
        { real_type{ 1.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.0 } }
    };
    const std::vector<real_type> weights{ real_type{ 1.0 }, real_type{ -1.0 }, real_type{ 1.0 }, real_type{ -1.0 } };
    const real_type rho{ 0.0 };
    std::vector<real_type> w{};
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.0 }, real_type{ 1.0 }, real_type{ 1.0 }, real_type{ 1.0 } },
        { real_type{ 1.0 }, real_type{ -1.0 }, real_type{ 1.0 }, real_type{ -1.0 } }
    };

    // create C-SVM: must be done using the mock class, since solve_system_of_linear_equations_impl is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    // predict the values using the previously learned support vectors and weights
    const std::vector<real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.size(), data.size());
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, (std::vector<real_type>{ real_type{ 0.0 }, real_type{ 4.0 } }));
    // in case of the linear kernel, the w vector should have been filled
    if (kernel == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.size(), support_vectors.front().size());
        EXPECT_FLOATING_POINT_VECTOR_NEAR(w, weights);
    } else {
        EXPECT_TRUE(w.empty());
    }
}

TYPED_TEST_P(GenericCSVM, predict) {
    using csvm_type = typename TypeParam::csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<real_type> test_data{ PLSSVM_TEST_PATH "/data/predict/500x200_test.libsvm" };

    // read the previously learned model
    const plssvm::model<real_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/500x200_{}.libsvm.model", kernel) };

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(params, TypeParam::additional_arguments);

    // predict label
    const std::vector<int> calculated = svm.predict(model, test_data);

    // read ground truth from file
    std::ifstream prediction_file{ PLSSVM_TEST_PATH "/data/predict/500x200.libsvm.predict" };
    const std::vector<int> ground_truth{ std::istream_iterator<int>{ prediction_file }, std::istream_iterator<int>{} };

    // check the calculated result for correctness
    EXPECT_EQ(calculated, ground_truth);
}

TYPED_TEST_P(GenericCSVM, score) {
    using csvm_type = typename TypeParam::csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<real_type> test_data{ PLSSVM_TEST_PATH "/data/predict/500x200_test.libsvm" };

    // read the previously learned model
    const plssvm::model<real_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/500x200_{}.libsvm.model", kernel) };

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(params, TypeParam::additional_arguments);

    // predict label
    const real_type calculated = svm.score(model, test_data);

    // check the calculated result for correctness
    EXPECT_EQ(calculated, real_type{ 1.0 });
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVM,
                            solve_system_of_linear_equations_trivial, solve_system_of_linear_equations, solve_system_of_linear_equations_with_correction,
                            predict_values, predict, score);
// clang-format on

//*************************************************************************************************************************************//
//                                                           CSVM DeathTests                                                           //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMDeathTest : public GenericCSVM<T> {};
TYPED_TEST_SUITE_P(GenericCSVMDeathTest);

TYPED_TEST_P(GenericCSVMDeathTest, solve_system_of_linear_equations) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::solve_system_of_linear_equations_impl is protected
    plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel };
    if (params.kernel_type != plssvm::kernel_function_type::linear) {
        params.gamma = real_type{ 0.1 };
    }
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    const std::vector<real_type> b{ real_type{ 1.0 }, real_type{ 2.0 } };

    // empty data is not allowed
    EXPECT_DEATH(std::ignore = svm.solve_system_of_linear_equations(params, std::vector<std::vector<real_type>>{}, b, real_type{ 0.1 }, 2),
                 "The data must not be empty!");
    // empty features are not allowed
    EXPECT_DEATH(std::ignore = (svm.solve_system_of_linear_equations(params, std::vector<std::vector<real_type>>{ std::vector<real_type>{} }, b, real_type{ 0.1 }, 2)),
                 "The data points must contain at least one feature!");
    // all data points must have the same number of features
    EXPECT_DEATH(std::ignore = (svm.solve_system_of_linear_equations(params, std::vector<std::vector<real_type>>{ std::vector<real_type>{ real_type{ 1.0 } }, std::vector<real_type>{ real_type{ 1.0 }, real_type{ 2.0 } } }, b, real_type{ 0.1 }, 2)),
                 "All data points must have the same number of features!");

    const std::vector<std::vector<real_type>> data = {
        { real_type{ 1.0 }, real_type{ 2.0 } },
        { real_type{ 3.0 }, real_type{ 4.0 } }
    };

    // the number of data points and values in b must be the same
    EXPECT_DEATH(std::ignore = svm.solve_system_of_linear_equations(params, data, std::vector<real_type>{}, 0.1, 2),
                 ::testing::HasSubstr("The number of data points in the matrix A (2) and the values in the right hand side vector (0) must be the same!"));
    // the stopping criterion must be greater than zero
    EXPECT_DEATH(std::ignore = svm.solve_system_of_linear_equations(params, data, b, real_type{ 0.0 }, 2),
                 "The stopping criterion in the CG algorithm must be greater than 0.0, but is 0!");
    EXPECT_DEATH(std::ignore = svm.solve_system_of_linear_equations(params, data, b, real_type{ -0.1 }, 2),
                 "The stopping criterion in the CG algorithm must be greater than 0.0, but is -0.1!");
    // at least one CG iteration must be performed
    EXPECT_DEATH(std::ignore = svm.solve_system_of_linear_equations(params, data, b, real_type{ 0.1 }, 0),
                 "The number of CG iterations must be greater than 0!");
}

TYPED_TEST_P(GenericCSVMDeathTest, predict_values) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::solve_system_of_linear_equations_impl is protected
    plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel };
    if (params.kernel_type != plssvm::kernel_function_type::linear) {
        params.gamma = real_type{ 0.1 };
    }
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    const std::vector<std::vector<real_type>> data = {
        { real_type{ 1.0 }, real_type{ 2.0 } },
        { real_type{ 3.0 }, real_type{ 4.0 } }
    };
    const std::vector<real_type> alpha{ real_type{ 1.0 }, real_type{ 2.0 } };
    std::vector<real_type> w{};

    // empty support vector data is not allowed
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, std::vector<std::vector<real_type>>{}, alpha, real_type{ 0.0 }, w, data)),
                 "The support vectors must not be empty!");
    // empty support vector features are not allowed
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, std::vector<std::vector<real_type>>{ std::vector<real_type>{} }, alpha, real_type{ 0.0 }, w, data)),
                 "The support vectors must contain at least one feature!");
    // all support vector must have the same number of features
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, std::vector<std::vector<real_type>>{ std::vector<real_type>{ real_type{ 1.0 } }, std::vector<real_type>{ real_type{ 1.0 }, real_type{ 2.0 } } }, alpha, real_type{ 0.0 }, w, data)),
                 "All support vectors must have the same number of features!");

    // number of support vectors and weights must be the same
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, data, std::vector<real_type>{ real_type{ 1.0 } }, real_type{ 0.0 }, w, data)),
                 ::testing::HasSubstr("The number of support vectors (2) and number of weights (1) must be the same!"));
    // either w must be empty or contain num features many entries
    w.resize(1);
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, data, alpha, real_type{ 0.0 }, w, data)),
                 ::testing::HasSubstr("Either w must be empty or contain exactly the same number of values (1) as features are present (2)!"));
    w.clear();

    // empty data points to predict is not allowed
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, data, alpha, real_type{ 0.0 }, w, std::vector<std::vector<real_type>>{})),
                 "The data points to predict must not be empty!");
    // empty data points to predict features are not allowed
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, data, alpha, real_type{ 0.0 }, w, std::vector<std::vector<real_type>>{ std::vector<real_type>{} })),
                 "The data points to predict must contain at least one feature!");
    // all data points to predict must have the same number of features
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, data, alpha, real_type{ 0.0 }, w, std::vector<std::vector<real_type>>{ std::vector<real_type>{ real_type{ 1.0 } }, std::vector<real_type>{ real_type{ 1.0 }, real_type{ 2.0 } } })),
                 "All data points to predict must have the same number of features!");
    // the number of features in the support vectors and data points to predict must be the same
    EXPECT_DEATH(std::ignore = (svm.predict_values(params, data, alpha, real_type{ 0.0 }, w, std::vector<std::vector<real_type>>{ std::vector<real_type>{ real_type{ 1.0 } }, std::vector<real_type>{ real_type{ 2.0 } } })),
                 ::testing::HasSubstr("The number of features in the support vectors (2) must be the same as in the data points to predict (1)!"));
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMDeathTest,
                            solve_system_of_linear_equations, predict_values);
// clang-format on

//*************************************************************************************************************************************//
//                                                               GPUCSVM                                                               //
//*************************************************************************************************************************************//

template <typename T>
class GenericGPUCSVM : public ::testing::Test, protected util::redirect_output<> {};
TYPED_TEST_SUITE_P(GenericGPUCSVM);

TYPED_TEST_P(GenericGPUCSVM, generate_q) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel, 2, 0.001, 1.0, 0.1 };

    // create the data that should be used
    const plssvm::data_set<real_type> data{ PLSSVM_TEST_FILE };

    // calculate correct q vector (ground truth)
    const std::vector<real_type> ground_truth = compare::generate_q(params, data.data());

    // create C-SVM: must be done using the mock class, since plssvm::openmp::csvm::generate_q is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    // perform the data setup on the device
    constexpr std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    const std::size_t num_used_devices = svm.select_num_used_devices(params.kernel_type, data.num_features());
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(data.data(), data.num_data_points() - 1, data.num_features(), boundary_size, num_used_devices);

    // calculate the q vector using a GPU backend
    const std::vector<real_type> calculated = svm.generate_q(params, data_d, data_last_d, data.num_data_points() - 1, feature_ranges, boundary_size);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, ground_truth);
}

TYPED_TEST_P(GenericGPUCSVM, calculate_w) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;
    using queue_type = typename mock_csvm_type::queue_type;

    // create the data that should be used
    const plssvm::data_set<real_type> support_vectors{ PLSSVM_TEST_FILE };
    const std::vector<real_type> weights = util::generate_random_vector<real_type>(support_vectors.num_data_points());

    // calculate the correct w vector
    const std::vector<real_type> ground_truth = compare::calculate_w(support_vectors.data(), weights);

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::calculate_w is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // perform the data setup on the device
    constexpr std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    const std::size_t num_support_vectors = support_vectors.num_data_points();
    const std::size_t num_used_devices = svm.select_num_used_devices(plssvm::kernel_function_type::linear, support_vectors.num_features());
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(support_vectors.data(), num_support_vectors - 1, support_vectors.num_features(), boundary_size, num_used_devices);

    std::vector<device_ptr_type> alpha_d(num_used_devices);
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        alpha_d[device] = device_ptr_type{ num_support_vectors + plssvm::THREAD_BLOCK_SIZE, svm.devices_[device] };
        alpha_d[device].memset(0);
        alpha_d[device].copy_to_device(weights, 0, num_support_vectors);
    }

    // calculate the w vector using a GPU backend
    const std::vector<real_type> calculated = svm.calculate_w(data_d, data_last_d, alpha_d, num_support_vectors, feature_ranges);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(calculated, ground_truth, real_type{ 1.0e6 });
}

TYPED_TEST_P(GenericGPUCSVM, run_device_kernel) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;
    using queue_type = typename mock_csvm_type::queue_type;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel, 2, 0.001, 1.0, 0.1 };

    // create the data that should be used
    const plssvm::data_set<real_type> data{ PLSSVM_TEST_FILE };
    const std::size_t dept = data.num_data_points() - 1;
    const std::vector<real_type> rhs = util::generate_random_vector<real_type>(dept, real_type{ 1.0 }, real_type{ 2.0 });
    const std::vector<real_type> q = compare::generate_q(params, data.data());
    const real_type QA_cost = compare::kernel_function(params, data.data().back(), data.data().back()) + 1 / params.cost;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::calculate_w is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    // perform the data setup on the device
    constexpr std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    const std::size_t num_used_devices = svm.select_num_used_devices(kernel, data.num_features());
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(data.data(), dept, data.num_features(), boundary_size, num_used_devices);
    std::vector<device_ptr_type> q_d{};
    std::vector<device_ptr_type> x_d{};
    std::vector<device_ptr_type> r_d{};

    for (const queue_type &queue : svm.devices_) {
        q_d.emplace_back(dept + boundary_size, queue).copy_to_device(q, 0, dept);
        x_d.emplace_back(dept + boundary_size, queue).memset(0);
        x_d.back().copy_to_device(rhs, 0, dept);
        r_d.emplace_back(dept + boundary_size, queue).memset(0);
    }

    for (const real_type add : { real_type{ -1.0 }, real_type{ 1.0 } }) {
        // calculate the correct device function result
        const std::vector<real_type> ground_truth = compare::device_kernel_function(params, data.data(), rhs, q, QA_cost, add);

        // perform the kernel calculation on the device
        for (std::size_t device = 0; device < num_used_devices; ++device) {
            svm.run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], data_d[device], feature_ranges, QA_cost, add, dept, boundary_size);
        }
        std::vector<real_type> calculated(dept);
        svm.device_reduction(r_d, calculated);

        for (device_ptr_type &r_d_device : r_d) {
            r_d_device.memset(0);
        }

        // check the calculated result for correctness
        EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, ground_truth);
    }
}

TYPED_TEST_P(GenericGPUCSVM, device_reduction) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    using namespace plssvm::operators;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;
    using queue_type = typename mock_csvm_type::queue_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::device_reduction is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // data to reduce
    const std::vector<real_type> data(1'024, real_type{ 1.0 });

    // perform the kernel calculation on the device
    std::vector<device_ptr_type> data_d{};
    for (const queue_type &queue : svm.devices_) {
        data_d.emplace_back(data.size(), queue).copy_to_device(data);
    }

    std::vector<real_type> calculated(data.size());
    svm.device_reduction(data_d, calculated);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, data * static_cast<real_type>(svm.devices_.size()));

    // check if the values have been correctly promoted back onto the device(s)
    for (std::size_t device = 0; device < svm.devices_.size(); ++device) {
        std::vector<real_type> device_data(data.size());
        data_d[device].copy_to_host(device_data);
        EXPECT_FLOATING_POINT_VECTOR_NEAR(device_data, data * static_cast<real_type>(svm.devices_.size()));
    }

    // test reduction of a single device
    data_d.resize(1);
    calculated.assign(data.size(), real_type{ 0.0 });
    svm.device_reduction(data_d, calculated);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, data);
    std::vector<real_type> device_data(data.size());
    data_d.front().copy_to_host(device_data);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(device_data, data);
}

TYPED_TEST_P(GenericGPUCSVM, select_num_used_devices) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::select_num_used_devices is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    if (kernel == plssvm::kernel_function_type::linear) {
        // the linear kernel function is compatible with at least one device
        EXPECT_GE(svm.select_num_used_devices(kernel, 1'204), 1);
    } else {
        // the other kernel functions are compatible only with exactly one device
        EXPECT_EQ(svm.select_num_used_devices(kernel, 1'204), 1);
    }
    // if only a single feature is provided, only a single device may ever be used
    EXPECT_EQ(svm.select_num_used_devices(kernel, 1), 1);
}

TYPED_TEST_P(GenericGPUCSVM, setup_data_on_device_minimal) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    // minimal example
    const std::vector<std::vector<real_type>> input = {
        { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } },
        { real_type{ 7.0 }, real_type{ 8.0 }, real_type{ 9.0 } },
    };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // perform data setup on the device
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(input, input.size(), input.front().size(), 0, 1);

    // check returned values
    // the "big" data vector
    ASSERT_EQ(data_d.size(), 1);
    ASSERT_EQ(data_d[0].size(), 9);
    std::vector<real_type> data(9);
    data_d[0].copy_to_host(data);
    EXPECT_FLOATING_POINT_VECTOR_EQ(data, (std::vector<real_type>{ real_type{ 1.0 }, real_type{ 4.0 }, real_type{ 7.0 }, real_type{ 2.0 }, real_type{ 5.0 }, real_type{ 8.0 }, real_type{ 3.0 }, real_type{ 6.0 }, real_type{ 9.0 } }));
    // the last row in the original data
    ASSERT_EQ(data_last_d.size(), 1);
    ASSERT_EQ(data_last_d[0].size(), 3);
    std::vector<real_type> data_last(3);
    data_last_d[0].copy_to_host(data_last);
    EXPECT_FLOATING_POINT_VECTOR_EQ(data_last, (std::vector<real_type>{ real_type{ 7.0 }, real_type{ 8.0 }, real_type{ 9.0 } }));
    // the feature ranges
    ASSERT_EQ(feature_ranges.size(), 2);
    EXPECT_EQ(feature_ranges, (std::vector<std::size_t>{ 0, input.front().size() }));
}

TYPED_TEST_P(GenericGPUCSVM, setup_data_on_device) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    // minimal example
    const std::vector<std::vector<real_type>> input = {
        { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } },
        { real_type{ 7.0 }, real_type{ 8.0 }, real_type{ 9.0 } },
    };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // perform data setup on the device
    const std::size_t boundary_size = 2;
    const std::size_t num_devices = svm.select_num_used_devices(plssvm::kernel_function_type::linear, input.front().size());
    const std::size_t num_data_points = input.size();
    const std::size_t num_features = input.front().size();

    // perform setup
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(input, num_data_points - 1, num_features - 1, boundary_size, num_devices);

    // check returned values
    // the feature ranges
    ASSERT_EQ(feature_ranges.size(), num_devices + 1);
    for (std::size_t i = 0; i <= num_devices; ++i) {
        EXPECT_EQ(feature_ranges[i], i * (num_features - 1) / num_devices);
    }

    const std::vector<real_type> transformed_data = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, input, boundary_size, num_data_points - 1);

    // the "big" data vector
    ASSERT_EQ(data_d.size(), num_devices);
    for (std::size_t device = 0; device < num_devices; ++device) {
        const std::size_t expected_size = (feature_ranges[device + 1] - feature_ranges[device]) * (num_data_points - 1 + boundary_size);
        ASSERT_EQ(data_d[device].size(), expected_size) << fmt::format("for device {}: [{}, {}]", device, feature_ranges[device], feature_ranges[device + 1]);
        std::vector<real_type> data(expected_size);
        data_d[device].copy_to_host(data);

        const std::size_t device_data_size = (feature_ranges[device + 1] - feature_ranges[device]) * (num_data_points - 1 + boundary_size);
        const std::vector<real_type> ground_truth(transformed_data.data() + feature_ranges[device] * (num_data_points - 1 + boundary_size),
                                                  transformed_data.data() + feature_ranges[device] * (num_data_points - 1 + boundary_size) + device_data_size);
        EXPECT_FLOATING_POINT_VECTOR_EQ(data, ground_truth);
    }
    // the last row in the original data
    ASSERT_EQ(data_last_d.size(), num_devices);
    for (std::size_t device = 0; device < num_devices; ++device) {
        const std::size_t expected_size = feature_ranges[device + 1] - feature_ranges[device] + boundary_size;
        ASSERT_EQ(data_last_d[device].size(), expected_size);
        std::vector<real_type> data_last(expected_size);
        data_last_d[device].copy_to_host(data_last);

        std::vector<real_type> ground_truth(input.back().data() + feature_ranges[device], input.back().data() + feature_ranges[device] + (feature_ranges[device + 1] - feature_ranges[device]));
        ground_truth.resize(ground_truth.size() + boundary_size, real_type{ 0.0 });
        EXPECT_FLOATING_POINT_VECTOR_EQ(data_last, ground_truth);
    }
}

TYPED_TEST_P(GenericGPUCSVM, num_available_devices) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::devices_ is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // test the number of available devices
    ASSERT_EQ(svm.num_available_devices(), svm.devices_.size());
    EXPECT_GE(svm.num_available_devices(), 1);
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVM,
                            generate_q, calculate_w, run_device_kernel, device_reduction,
                            select_num_used_devices, setup_data_on_device_minimal, setup_data_on_device, num_available_devices);
// clang-format on

//*************************************************************************************************************************************//
//                                                         GPUCSVM DeathTests                                                          //
//*************************************************************************************************************************************//

template <typename T>
class GenericGPUCSVMDeathTest : public GenericGPUCSVM<T> {};
TYPED_TEST_SUITE_P(GenericGPUCSVMDeathTest);

TYPED_TEST_P(GenericGPUCSVMDeathTest, generate_q) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel, 2, 0.001, 1.0, 0.1 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::generate_q is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    // valid feature_range
    const std::vector<std::size_t> feature_range{ 0, 1 };

    // the data_d vector must not be empty!
    std::vector<device_ptr_type> vec1;
    vec1.emplace_back(1, svm.devices_[0]);
    std::vector<device_ptr_type> vec2;
    vec2.emplace_back(1, svm.devices_[0]);
    vec2.emplace_back(1, svm.devices_[0]);

    EXPECT_DEATH(std::ignore = svm.generate_q(params, std::vector<device_ptr_type>{}, vec1, 1, feature_range, 0),
                 "The data_d array may not be empty!");
    // the ptr in the data_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.generate_q(params, std::vector<device_ptr_type>(1), vec1, 1, feature_range, 0),
                 "Each device_ptr in data_d must at least contain one data point!");
    // the data_last_d vector must not be empty!
    EXPECT_DEATH(std::ignore = svm.generate_q(params, vec1, std::vector<device_ptr_type>{}, 1, feature_range, 0),
                 "The data_last_d array may not be empty!");
    // the ptr in the data_last_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.generate_q(params, vec1, std::vector<device_ptr_type>(1), 1, feature_range, 0),
                 "Each device_ptr in data_last_d must at least contain one data point!");
    // data_d and data_last_d must have the same size
    EXPECT_DEATH(std::ignore = svm.generate_q(params, vec1, vec2, 1, feature_range, 0),
                 "The number of used devices to the data_d and data_last_d vectors must be equal!: 1 != 2");

    // the number of data points must be greater than zero
    EXPECT_DEATH(std::ignore = svm.generate_q(params, vec1, vec1, 0, feature_range, 0),
                 "At least one data point must be used to calculate q!");
    // the number of elements in the feature_ranges vector must be one greater than the number of used devices
    EXPECT_DEATH(std::ignore = svm.generate_q(params, vec1, vec1, 1, std::vector<std::size_t>{}, 0),
                 ::testing::HasSubstr("The number of values in the feature_range vector must be exactly one more than the number of used devices!: 0 != 1 + 1"));
    // the values in the feature_ranges vector must be monotonically increasing
    EXPECT_DEATH(std::ignore = svm.generate_q(params, vec1, vec1, 1, { 1, 0 }, 0),
                 "The feature ranges are not monotonically increasing!");
}

TYPED_TEST_P(GenericGPUCSVMDeathTest, calculate_w) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::generate_q is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // valid feature_range
    const std::vector<std::size_t> feature_range{ 0, 1 };

    // the data_d vector must not be empty!
    std::vector<device_ptr_type> vec1;
    vec1.emplace_back(1, svm.devices_[0]);
    std::vector<device_ptr_type> vec2;
    vec2.emplace_back(1, svm.devices_[0]);
    vec2.emplace_back(1, svm.devices_[0]);

    // the data_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.calculate_w(std::vector<device_ptr_type>{}, vec1, vec1, 1, feature_range),
                 "The data_d array may not be empty!");
    // the ptr in the data_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.calculate_w(std::vector<device_ptr_type>(1), vec1, vec1, 1, feature_range),
                 "Each device_ptr in data_d must at least contain one data point!");
    // the data_last_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, std::vector<device_ptr_type>{}, vec1, 1, feature_range),
                 "The data_last_d array may not be empty!");
    // the ptr in the data_last_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, std::vector<device_ptr_type>(1), vec1, 1, feature_range),
                 "Each device_ptr in data_last_d must at least contain one data point!");
    // data_d and data_last_d must have the same size
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec2, vec1, 1, feature_range),
                 "The number of used devices to the data_d and data_last_d vectors must be equal!: 1 != 2");
    // the alpha_d vector must not be empty!
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec1, std::vector<device_ptr_type>{}, 1, feature_range),
                 "The alpha_d array may not be empty!");
    // the ptr in the alpha_d vector must not be empty
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec1, std::vector<device_ptr_type>(1), 1, feature_range),
                 "Each device_ptr in alpha_d must at least contain one data point!");
    // data_d and alpha_d must have the same size
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec1, vec2, 1, feature_range),
                 "The number of used devices to the data_d and alpha_d vectors must be equal!: 1 != 2");

    // the number of data points must be greater than zero
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec1, vec1, 0, feature_range),
                 "At least one data point must be used to calculate q!");
    // the number of elements in the feature_ranges vector must be one greater than the number of used devices
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec1, vec1, 1, std::vector<std::size_t>{}),
                 ::testing::HasSubstr("The number of values in the feature_range vector must be exactly one more than the number of used devices!: 0 != 1 + 1"));
    // the values in the feature_ranges vector must be monotonically increasing
    EXPECT_DEATH(std::ignore = svm.calculate_w(vec1, vec1, vec1, 1, { 1, 0 }),
                 "The feature ranges are not monotonically increasing!");
}

TYPED_TEST_P(GenericGPUCSVMDeathTest, run_device_kernel) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel, 2, 0.001, 1.0, 0.1 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::run_device_kernel is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, TypeParam::additional_arguments);

    // valid feature_range
    const std::vector<std::size_t> feature_range{ 0, 1 };

    // the data_d vector must not be empty!
    device_ptr_type ptr{ 1, svm.devices_[0] };
    device_ptr_type empty_ptr{};

    // the device id must be smaller than the total number of available devices
    EXPECT_DEATH(svm.run_device_kernel(svm.num_available_devices(), params, ptr, ptr, ptr, ptr, feature_range, real_type{ 0.0 }, real_type{ 1.0 }, 1, 0),
                 ::testing::HasSubstr(fmt::format("Requested device {}, but only {} device(s) are available!", svm.num_available_devices(), svm.num_available_devices())));
    // the q_d device_ptr must not be empty
    EXPECT_DEATH(svm.run_device_kernel(0, params, empty_ptr, ptr, ptr, ptr, feature_range, real_type{ 0.0 }, real_type{ 1.0 }, 1, 0),
                 "The q_d device_ptr may not be empty!");
    // the r_d device_ptr must not be empty
    EXPECT_DEATH(svm.run_device_kernel(0, params, ptr, empty_ptr, ptr, ptr, feature_range, real_type{ 0.0 }, real_type{ 1.0 }, 1, 0),
                 "The r_d device_ptr may not be empty!");
    // the x_d device_ptr must not be empty
    EXPECT_DEATH(svm.run_device_kernel(0, params, ptr, ptr, empty_ptr, ptr, feature_range, real_type{ 0.0 }, real_type{ 1.0 }, 1, 0),
                 "The x_d device_ptr may not be empty!");
    // the data_d device_ptr must not be empty
    EXPECT_DEATH(svm.run_device_kernel(0, params, ptr, ptr, ptr, empty_ptr, feature_range, real_type{ 0.0 }, real_type{ 1.0 }, 1, 0),
                 "The data_d device_ptr may not be empty!");

    // the values in the feature_ranges vector must be monotonically increasing
    EXPECT_DEATH(svm.run_device_kernel(0, params, ptr, ptr, ptr, ptr, { 1, 0 }, real_type{ 0.0 }, real_type{ 0.0 }, 1, 0),
                 "The feature ranges are not monotonically increasing!");

    // add must either be -1.0 or 1.0
    EXPECT_DEATH(svm.run_device_kernel(0, params, ptr, ptr, ptr, ptr, feature_range, real_type{ 0.0 }, real_type{ 0.0 }, 1, 0),
                 "add must either by -1.0 or 1.0, but is 0!");
    // at least one data point must be available
    EXPECT_DEATH(svm.run_device_kernel(0, params, ptr, ptr, ptr, ptr, feature_range, real_type{ 0.0 }, real_type{ 1.0 }, 0, 0),
                 "At least one data point must be used to calculate q!");
}

TYPED_TEST_P(GenericGPUCSVMDeathTest, device_reduction) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::generate_q is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // the data_d vector must not be empty!
    std::vector<device_ptr_type> vec1;
    vec1.emplace_back(1, svm.devices_[0]);

    std::vector<device_ptr_type> empty_device_ptr_vec{};
    std::vector<device_ptr_type> one_element_device_ptr_vec(1);
    std::vector<real_type> empty_buffer_vec{};
    std::vector<real_type> one_element_buffer_vec(1);

    // the buffer_d vector must not be empty
    EXPECT_DEATH(svm.device_reduction(empty_device_ptr_vec, one_element_buffer_vec),
                 "The buffer_d array may not be empty!");
    // the ptr in the buffer_d vector must not be empty
    EXPECT_DEATH(svm.device_reduction(one_element_device_ptr_vec, one_element_buffer_vec),
                 "Each device_ptr in buffer_d must at least contain one data point!");
    // the buffer vector must not be empty
    EXPECT_DEATH(svm.device_reduction(vec1, empty_buffer_vec),
                 "The buffer array may not be empty!");
}

TYPED_TEST_P(GenericGPUCSVMDeathTest, select_num_used_devices) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = TypeParam::kernel_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::select_num_used_devices is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // at least one feature must be provided
    EXPECT_DEATH(std::ignore = svm.select_num_used_devices(kernel, 0), "At lest one feature must be given!");
}

TYPED_TEST_P(GenericGPUCSVMDeathTest, setup_data_on_device) {
    using mock_csvm_type = typename TypeParam::mock_csvm_type;
    using real_type = typename TypeParam::real_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(TypeParam::additional_arguments);

    // empty data is not allowed
    EXPECT_DEATH(std::ignore = svm.setup_data_on_device(std::vector<std::vector<real_type>>{}, 1, 1, 0, 1), "The data must not be empty!");
    // empty features are not allowed
    EXPECT_DEATH(std::ignore = (svm.setup_data_on_device(std::vector<std::vector<real_type>>{ std::vector<real_type>{} }, 1, 1, 0, 1)), "The data points must contain at least one feature!");
    // all data points must have the same number of features
    EXPECT_DEATH(std::ignore = (svm.setup_data_on_device(std::vector<std::vector<real_type>>{ std::vector<real_type>{ real_type{ 1.0 } }, std::vector<real_type>{ real_type{ 1.0 }, real_type{ 2.0 } } }, 1, 1, 0, 1)),
                 "All data points must have the same number of features!");

    const std::vector<std::vector<real_type>> data = {
        { real_type{ 1.0 }, real_type{ 2.0 } },
        { real_type{ 3.0 }, real_type{ 4.0 } }
    };

    // at least one data point must be copied to the device
    EXPECT_DEATH(std::ignore = svm.setup_data_on_device(data, 0, 1, 0, 1), "At least one data point must be copied to the device!");
    // at most two data point can be copied to the device
    EXPECT_DEATH(std::ignore = svm.setup_data_on_device(data, 3, 1, 0, 1), "Can't copy more data points to the device than are present!: 3 <= 2");
    // at least one feature must be copied to the device
    EXPECT_DEATH(std::ignore = svm.setup_data_on_device(data, 1, 0, 0, 1), "At least one feature must be copied to the device!");
    // at most two features can be copied to the device
    EXPECT_DEATH(std::ignore = svm.setup_data_on_device(data, 1, 3, 0, 1), "Can't copy more features to the device than are present!: 3 <= 2");

    // can't use more than the available device
    EXPECT_DEATH(std::ignore = svm.setup_data_on_device(data, 1, 1, 0, svm.num_available_devices() + 1),
                 fmt::format("Can't use more devices than are available!: {} <= {}", svm.num_available_devices() + 1, svm.num_available_devices()));
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericGPUCSVMDeathTest,
                            generate_q, calculate_w, run_device_kernel, device_reduction, select_num_used_devices, setup_data_on_device);
// clang-format on

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_TESTS_HPP_
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic C-SVR tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVR_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVR_TESTS_HPP_
#pragma once

#include "plssvm/constants.hpp"                     // plssvm::real_type
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/kernel_function_types.hpp"         // plssvm::kernel_function_type
#include "plssvm/model/regression_model.hpp"        // plssvm::regression_model
#include "plssvm/parameter.hpp"                     // plssvm::parameter
#include "plssvm/solver_types.hpp"                  // plssvm::solver_type
#include "plssvm/target_platforms.hpp"              // plssvm::target_platform

#include "tests/types_to_test.hpp"  // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"        // util::{redirect_output, construct_from_tuple}

#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, ::testing::Test

#include <utility>  // std::move
#include <vector>   // std::vector

template <typename T>
class GenericCSVR : public ::testing::Test,
                    protected util::redirect_output<> { };

TYPED_TEST_SUITE_P(GenericCSVR);

TYPED_TEST_P(GenericCSVR, move_constructor) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvr_type = typename csvm_test_type::csvr_type;

    // create normal C-SVR
    csvr_type svr{};

    // get current state
    const plssvm::parameter params = svr.get_params();
    const plssvm::target_platform target = svr.get_target_platform();

    // move construct new C-SVR
    const csvr_type new_svr{ std::move(svr) };

    // check that the state of the newly constructed C-SVC matches the old state of the moved-from C-SVC
    EXPECT_EQ(new_svr.get_params(), params);
    EXPECT_EQ(new_svr.get_target_platform(), target);
}

TYPED_TEST_P(GenericCSVR, move_assignment) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvr_type = typename csvm_test_type::csvr_type;

    // create normal C-SVR
    csvr_type svr{};

    // get current state
    const plssvm::parameter params = svr.get_params();
    const plssvm::target_platform target = svr.get_target_platform();

    // construct new C-SVR with a non-default state
    csvr_type new_svr{ plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial } };

    // move assign old C-SVC to the new one
    new_svr = std::move(svr);

    // check that the state of the newly constructed C-SVC matches the old state of the moved-from C-SVC
    EXPECT_EQ(new_svr.get_params(), params);
    EXPECT_EQ(new_svr.get_target_platform(), target);
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVR,
                            move_constructor,
                            move_assignment);

//*************************************************************************************************************************************//
//                                            C-SVR tests depending on the kernel function                                             //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVRKernelFunction : public GenericCSVR<T> { };

TYPED_TEST_SUITE_P(GenericCSVRKernelFunction);

TYPED_TEST_P(GenericCSVRKernelFunction, predict) {
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvr_type = typename csvm_test_type::csvr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::cost = 1000.0, plssvm::kernel_type = kernel };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
        params.gamma = 1.0;
    }
    if constexpr (kernel == plssvm::kernel_function_type::sigmoid) {
        params.gamma = 0.01;
    }

    // create data set that is always classifiable
    plssvm::regression_data_set<label_type> test_data = util::generate_trivially_solvable_regression_data_set<label_type>();
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (test_data.labels().has_value()) {
            test_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(test_data.data()), *test_data.labels() };
        }
    }

    // create normal C-SVR
    const csvr_type svr = util::construct_from_tuple<csvr_type>(params, csvm_test_type::additional_arguments);

    // fitting the test data will ALWAYS score 100% accuracy
    const plssvm::regression_model<label_type> model = svr.fit(test_data, plssvm::epsilon = 1e-16);

    // actual TEST: predict label
    std::vector<label_type> calculated = svr.predict(model, test_data);

    // check the calculated result for correctness
    if constexpr (std::is_floating_point_v<label_type>) {
        // convert a floating point label_type back to a plain integer
        for (label_type &val : calculated) {
            val = static_cast<label_type>(std::round(val));
        }
    }
    EXPECT_EQ(calculated, test_data.labels().value().get());
}

TYPED_TEST_P(GenericCSVRKernelFunction, score_model) {
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvr_type = typename csvm_test_type::csvr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create data set that is always classifiable
    plssvm::regression_data_set<label_type> test_data = util::generate_trivially_solvable_regression_data_set<label_type>();
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (test_data.labels().has_value()) {
            test_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(test_data.data()), *test_data.labels() };
        }
    }

    // create normal C-SVR
    const csvr_type svr = util::construct_from_tuple<csvr_type>(params, csvm_test_type::additional_arguments);

    // fitting the test data will ALWAYS score 100% accuracy
    const plssvm::regression_model<label_type> model = svr.fit(test_data, plssvm::epsilon = 1e-16);

    // actual TEST: score model
    [[maybe_unused]] const plssvm::real_type calculated = svr.score(model);

    // check the calculated result for correctness
    // 1.0 is the maximum possible value
    // arbitrary small (negative) values are possible, but the "easy" data set shouldn't result in values smaller 0.0
    EXPECT_INCLUSIVE_RANGE(calculated, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 });
}

TYPED_TEST_P(GenericCSVRKernelFunction, score) {
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvr_type = typename csvm_test_type::csvr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create data set that is always classifiable
    plssvm::regression_data_set<label_type> test_data = util::generate_trivially_solvable_regression_data_set<label_type>();
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (test_data.labels().has_value()) {
            test_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(test_data.data()), *test_data.labels() };
        }
    }

    // create normal C-SVR
    const csvr_type svr = util::construct_from_tuple<csvr_type>(params, csvm_test_type::additional_arguments);

    // fitting the test data will ALWAYS score 100% accuracy
    const plssvm::regression_model<label_type> model = svr.fit(test_data, plssvm::epsilon = 1e-16);

    // actual TEST:: score the test data using the learned model
    [[maybe_unused]] const plssvm::real_type calculated = svr.score(model, test_data);

    // check the calculated result for correctness
    // 1.0 is the maximum possible value
    // arbitrary small (negative) values are possible, but the "easy" data set shouldn't result in values smaller 0.0
    EXPECT_INCLUSIVE_RANGE(calculated, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 });
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVRKernelFunction,
                            predict,
                            score_model,
                            score);

//*************************************************************************************************************************************//
//                                       C-SVR tests depending on the solver and kernel function                                       //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVRSolverKernelFunction : public GenericCSVR<T> { };

TYPED_TEST_SUITE_P(GenericCSVRSolverKernelFunction);

TYPED_TEST_P(GenericCSVRSolverKernelFunction, fit) {
    // note: only quantitative tests, doesn't check the real weights and rho values
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvr_type = typename csvm_test_type::csvr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    plssvm::regression_data_set<label_type> test_data{ PLSSVM_TEST_PATH "/data/libsvm/regression/6x4.libsvm" };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (test_data.labels().has_value()) {
            test_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(test_data.data()), *test_data.labels() };
        }
    }

    // create normal C-SVR
    // const csvr_type svr = util::construct_from_tuple<csvr_type>(params, csvm_test_type::additional_arguments);
    const csvr_type svr{ params };

    // call fit
    const plssvm::regression_model<label_type> model = svr.fit(test_data, plssvm::epsilon = 1e-10, plssvm::solver = solver);

    // check the calculated result for correctness
    EXPECT_EQ(model.num_support_vectors(), test_data.num_data_points());
    EXPECT_EQ(model.num_features(), test_data.num_features());
    EXPECT_EQ(model.get_params(), (plssvm::parameter{ params, plssvm::gamma = plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(test_data.num_features()) }));
    EXPECT_EQ(model.support_vectors(), test_data.data());
    EXPECT_EQ(model.labels().value().get(), test_data.labels().value().get());
    EXPECT_EQ(model.weights().size(), 1);
    EXPECT_EQ(model.rho().size(), 1);
    EXPECT_TRUE(model.num_iters().has_value());
    EXPECT_EQ(model.num_iters().value().size(), 1);
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVRSolverKernelFunction,
                            fit);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVR_TESTS_HPP_

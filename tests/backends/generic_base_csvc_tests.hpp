/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic C-SVC tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVC_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVC_TESTS_HPP_
#pragma once

#include "plssvm/classification_types.hpp"              // plssvm::classification_type
#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/kernel_function_types.hpp"             // plssvm::kernel_function_type
#include "plssvm/model/classification_model.hpp"        // plssvm::classification_model
#include "plssvm/parameter.hpp"                         // plssvm::parameter
#include "plssvm/solver_types.hpp"                      // plssvm::solver_type
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform

#include "tests/types_to_test.hpp"  // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"        // util::{redirect_output, construct_from_tuple, temporary_file, instantiate_template_file}
#include "tests/custom_test_macros.hpp" // EXPECT_OPTIONAL_EQ

#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, ::testing::Test

#include <utility>  // std::move
#include <vector>   // std::vector

template <typename T>
class GenericCSVC : public ::testing::Test,
                    protected util::redirect_output<> { };

TYPED_TEST_SUITE_P(GenericCSVC);

TYPED_TEST_P(GenericCSVC, MoveConstructor) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvc_type = typename csvm_test_type::csvc_type;

    // create normal C-SVC
    csvc_type svc{};

    // get current state
    const plssvm::parameter params = svc.get_params();
    const plssvm::target_platform target = svc.get_target_platform();

    // move construct new C-SVC
    const csvc_type new_svc{ std::move(svc) };

    // check that the state of the newly constructed C-SVC matches the old state of the moved-from C-SVC
    EXPECT_EQ(new_svc.get_params(), params);
    EXPECT_EQ(new_svc.get_target_platform(), target);
}

TYPED_TEST_P(GenericCSVC, MoveAssignment) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvc_type = typename csvm_test_type::csvc_type;

    // create normal C-SVC
    csvc_type svc{};

    // get current state
    const plssvm::parameter params = svc.get_params();
    const plssvm::target_platform target = svc.get_target_platform();

    // construct new C-SVC with a non-default state
    csvc_type new_svc{ plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial } };

    // move assign old C-SVC to the new one
    new_svc = std::move(svc);

    // check that the state of the newly constructed C-SVC matches the old state of the moved-from C-SVC
    EXPECT_EQ(new_svc.get_params(), params);
    EXPECT_EQ(new_svc.get_target_platform(), target);
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVC,
                            MoveConstructor,
                            MoveAssignment);

//*************************************************************************************************************************************//
//                                C-SVC tests depending on the kernel function and classification type                                 //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVCKernelFunctionClassification : public GenericCSVC<T> { };

TYPED_TEST_SUITE_P(GenericCSVCKernelFunctionClassification);

TYPED_TEST_P(GenericCSVCKernelFunctionClassification, Predict) {
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvc_type = typename csvm_test_type::csvc_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create data set that is always classifiable
    plssvm::classification_data_set<label_type> test_data = util::generate_trivially_solvable_classification_data_set<label_type>();  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto& labels_opt = test_data.labels();
        if (labels_opt.has_value()) {
            test_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(test_data.data()), labels_opt.value() };
        }
    }

    // create normal C-SVC
    const auto svc = util::construct_from_tuple<csvc_type>(params, csvm_test_type::additional_arguments);

    // fitting the test data will ALWAYS score 100% accuracy
    const plssvm::classification_model<label_type> model = svc.fit(test_data, plssvm::epsilon = 1e-16, plssvm::classification = classification);

    // actual TEST: predict label
    const std::vector<label_type> calculated = svc.predict(model, test_data);

    // check the calculated result for correctness
    EXPECT_OPTIONAL_EQ(calculated, test_data.labels());

    // for the linear kernel, predict again to check whether reusing the w vector works as intended
    if (kernel == plssvm::kernel_function_type::linear) {
        const std::vector<label_type> calculated_second = svc.predict(model, test_data);
        EXPECT_OPTIONAL_EQ(calculated_second, test_data.labels());
    }
}

TYPED_TEST_P(GenericCSVCKernelFunctionClassification, ScoreModel) {
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvc_type = typename csvm_test_type::csvc_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create data set that is always classifiable
    plssvm::classification_data_set<label_type> test_data = util::generate_trivially_solvable_classification_data_set<label_type>();  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto& labels_opt = test_data.labels();
        if (labels_opt.has_value()) {
            test_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(test_data.data()), labels_opt.value() };
        }
    }

    // create normal C-SVC
    const auto svc = util::construct_from_tuple<csvc_type>(params, csvm_test_type::additional_arguments);

    // fitting the test data will ALWAYS score 100% accuracy
    const plssvm::classification_model<label_type> model = svc.fit(test_data, plssvm::epsilon = 1e-16, plssvm::classification = classification);

    // actual TEST: score model
    const plssvm::real_type calculated = svc.score(model);

    // check the calculated result for correctness
    EXPECT_EQ(calculated, plssvm::real_type{ 1.0 });
}

TYPED_TEST_P(GenericCSVCKernelFunctionClassification, Score) {
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvc_type = typename csvm_test_type::csvc_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create data set that is always classifiable
    plssvm::classification_data_set<label_type> test_data = util::generate_trivially_solvable_classification_data_set<label_type>();  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto& labels_opt = test_data.labels();
        if (labels_opt.has_value()) {
            test_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(test_data.data()), labels_opt.value() };
        }
    }

    // create normal C-SVC
    const auto svc = util::construct_from_tuple<csvc_type>(params, csvm_test_type::additional_arguments);

    // fitting the test data will ALWAYS score 100% accuracy
    const plssvm::classification_model<label_type> model = svc.fit(test_data, plssvm::epsilon = 1e-16, plssvm::classification = classification);

    // actual TEST:: score the test data using the learned model
    const plssvm::real_type calculated = svc.score(model, test_data);

    // check the calculated result for correctness
    EXPECT_EQ(calculated, plssvm::real_type{ 1.0 });
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVCKernelFunctionClassification,
                            Predict,
                            ScoreModel,
                            Score);

//*************************************************************************************************************************************//
//                            C-SVC tests depending on the solver, kernel function, and classification type                            //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVCSolverKernelFunctionClassification : public GenericCSVC<T>,
                                                      protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<1, T>;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    }
};

TYPED_TEST_SUITE_P(GenericCSVCSolverKernelFunctionClassification);

TYPED_TEST_P(GenericCSVCSolverKernelFunctionClassification, Fit) {
    // note: only quantitative tests, doesn't check the real weights and rho values
    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvc_type = typename csvm_test_type::csvc_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<2, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create data set to be used
    plssvm::classification_data_set<label_type> test_data{ this->filename };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto& labels_opt = test_data.labels();
        if (labels_opt.has_value()) {
            test_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(test_data.data()), labels_opt.value() };
        }
    }

    // create normal C-SVC
    const auto svc = util::construct_from_tuple<csvc_type>(params, csvm_test_type::additional_arguments);

    // call fit
    const plssvm::classification_model<label_type> model = svc.fit(test_data, plssvm::epsilon = 1e-10, plssvm::solver = solver, plssvm::classification = classification);

    // check the calculated result for correctness
    EXPECT_EQ(model.num_support_vectors(), test_data.num_data_points());
    EXPECT_EQ(model.num_features(), test_data.num_features());
    EXPECT_EQ(model.get_params(), (plssvm::parameter{ params, plssvm::gamma = plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(test_data.num_features()) }));
    EXPECT_EQ(model.support_vectors(), test_data.data());
    EXPECT_OPTIONAL_EQ(model.labels(), test_data.labels());
    EXPECT_EQ(model.num_classes(), test_data.num_classes());
    EXPECT_OPTIONAL_EQ(model.classes(), test_data.classes());
    if constexpr (classification == plssvm::classification_type::oaa) {
        EXPECT_EQ(model.weights().size(), 1);
        EXPECT_EQ(model.rho().size(), test_data.num_classes());
    } else {
        EXPECT_EQ(model.weights().size(), plssvm::calculate_number_of_classifiers(classification, test_data.num_classes()));
        EXPECT_EQ(model.rho().size(), plssvm::calculate_number_of_classifiers(classification, test_data.num_classes()));
    }
    EXPECT_EQ(model.get_classification_type(), classification);
    const auto& num_iters_opt = model.num_iters();
    ASSERT_TRUE(num_iters_opt.has_value());
    if (num_iters_opt.has_value()) {
        EXPECT_EQ(num_iters_opt.value().size(), (plssvm::calculate_number_of_classifiers(classification, test_data.num_classes())));
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVCSolverKernelFunctionClassification,
                            Fit);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVC_TESTS_HPP_

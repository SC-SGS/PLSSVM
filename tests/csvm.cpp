/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base C-SVM functions through its mock class.
 */

#include "mock_csvm.hpp"                     // mock_csvm
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/model.hpp"                  // plssvm::model
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::{real_type_to_name, real_type_label_type_combination_to_name}
#include "types_to_test.hpp"       // util::{real_type_gtest, real_type_label_type_combination_gtest}
#include "utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_CALL

#include <iostream>   // std::clog
#include <sstream>    // std::stringstream
#include <streambuf>  // std::streambuf
#include <string>     // std::string
#include <tuple>      // std::ignore
#include <vector>     // std::vector

template <typename T>
class BaseCSVM : public ::testing::Test {};
TYPED_TEST_SUITE(BaseCSVM, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(BaseCSVM, default_construct_from_parameter) {
    using real_type = TypeParam;

    // create mock_csvm using the default parameter (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), plssvm::parameter<real_type>{});
}
TYPED_TEST(BaseCSVM, construct_from_parameter) {
    using real_type = TypeParam;

    // create parameter class
    const plssvm::parameter<real_type> params{ plssvm::kernel_function_type::polynomial, 4, real_type{ 0.2 }, real_type{ 0.1 }, real_type{ 0.01 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ params };

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TYPED_TEST(BaseCSVM, construct_from_parameter_invalid_kernel_type) {
    using real_type = TypeParam;

    // create parameter class
    plssvm::parameter<real_type> params{};
    params.kernel_type = static_cast<plssvm::kernel_function_type>(3);

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid kernel type must throw
    EXPECT_THROW_WHAT(mock_csvm<real_type>{ params }, plssvm::invalid_parameter_exception, "Invalid kernel function 3 given!");
}
TYPED_TEST(BaseCSVM, construct_from_parameter_invalid_gamma) {
    using real_type = TypeParam;

    // create parameter class
    plssvm::parameter<real_type> params{};
    params.kernel_type = plssvm::kernel_function_type::polynomial;
    params.gamma = -1.0;

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid value for gamma must throw
    EXPECT_THROW_WHAT(mock_csvm<real_type>{ params },
                      plssvm::invalid_parameter_exception,
                      "gamma must be greater than 0.0, but is -1!");
}

TYPED_TEST(BaseCSVM, construct_linear_from_named_parameters) {
    using real_type = TypeParam;

    // correct parameters
    plssvm::parameter<real_type> params{};
    params.cost = 2.0;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ plssvm::kernel_function_type::linear, plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}
TYPED_TEST(BaseCSVM, construct_polynomial_from_named_parameters) {
    using real_type = TypeParam;

    // correct parameters
    const plssvm::parameter<real_type> params{ plssvm::kernel_function_type::polynomial, 4, 0.1, 1.2, 0.001 };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ plssvm::kernel_function_type::polynomial,
                                     plssvm::degree = params.degree,
                                     plssvm::gamma = params.gamma,
                                     plssvm::coef0 = params.coef0,
                                     plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}
TYPED_TEST(BaseCSVM, construct_rbf_from_named_parameters) {
    using real_type = TypeParam;

    // correct parameters
    plssvm::parameter<real_type> params{};
    params.kernel_type = plssvm::kernel_function_type::rbf;
    params.gamma = 0.00001;
    params.cost = 10.0;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ plssvm::kernel_function_type::rbf,
                                     plssvm::gamma = params.gamma,
                                     plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

template <typename T>
class BaseCSVMWarning : public BaseCSVM<T> {
  public:
    void start_capturing() {
        sbuf = std::clog.rdbuf();
        std::clog.rdbuf(buffer.rdbuf());
    }
    void end_capturing() {
        std::clog.rdbuf(sbuf);
        sbuf = nullptr;
    }
    std::string get_capture() {
        return buffer.str();
    }

  private:
    std::stringstream buffer{};
    std::streambuf *sbuf{};
};
TYPED_TEST_SUITE(BaseCSVMWarning, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(BaseCSVMWarning, construct_unused_parameter_warning_degree) {
    using real_type = TypeParam;

    // start capture of std::clog
    this->start_capturing();

    const mock_csvm<real_type> csvm{ plssvm::kernel_function_type::linear, plssvm::degree = 2 };

    // end capture of std::clog
    this->end_capturing();

    EXPECT_EQ(this->get_capture(), "degree parameter provided, which is not used in the linear kernel (u'*v)!\n");
}
TYPED_TEST(BaseCSVMWarning, construct_unused_parameter_warning_gamma) {
    using real_type = TypeParam;

    // start capture of std::clog
    this->start_capturing();

    const mock_csvm<real_type> csvm{ plssvm::kernel_function_type::linear, plssvm::gamma = 0.1 };

    // end capture of std::clog
    this->end_capturing();

    EXPECT_EQ(this->get_capture(), "gamma parameter provided, which is not used in the linear kernel (u'*v)!\n");
}
TYPED_TEST(BaseCSVMWarning, construct_unused_parameter_warning_coef0) {
    using real_type = TypeParam;

    // start capture of std::clog
    this->start_capturing();

    const mock_csvm<real_type> csvm{ plssvm::kernel_function_type::linear, plssvm::coef0 = 0.1 };

    // end capture of std::clog
    this->end_capturing();

    EXPECT_EQ(this->get_capture(), "coef0 parameter provided, which is not used in the linear kernel (u'*v)!\n");
}

TYPED_TEST(BaseCSVM, construct_from_named_parameters_invalid_kernel_type) {
    using real_type = TypeParam;

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid kernel type must throw
    EXPECT_THROW_WHAT(mock_csvm<real_type>{ static_cast<plssvm::kernel_function_type>(3) },
                      plssvm::invalid_parameter_exception,
                      "Invalid kernel function 3 given!");
}
TYPED_TEST(BaseCSVM, construct_from_named_parameters_invalid_gamma) {
    using real_type = TypeParam;

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid value for gamma must throw
    EXPECT_THROW_WHAT((mock_csvm<real_type>{ plssvm::kernel_function_type::polynomial, plssvm::gamma = -1.0 }),
                      plssvm::invalid_parameter_exception,
                      "gamma must be greater than 0.0, but is -1!");
}

TYPED_TEST(BaseCSVM, get_params) {
    using real_type = TypeParam;

    // create parameter class
    const plssvm::parameter<real_type> params{ plssvm::kernel_function_type::polynomial, 4, real_type{ 0.2 }, real_type{ 0.1 }, real_type{ 0.01 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ params };

    // check whether the parameters have been set correctly
    const plssvm::parameter<real_type> csvm_params = csvm.get_params();
    EXPECT_EQ(csvm_params, params);
    EXPECT_TRUE(csvm_params.equivalent(params));
}

TYPED_TEST(BaseCSVM, set_params_from_parameter) {
    using real_type = TypeParam;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    mock_csvm<real_type> csvm{};

    // create parameter class
    const plssvm::parameter<real_type> params{ plssvm::kernel_function_type::polynomial, 4, real_type{ 0.2 }, real_type{ 0.1 }, real_type{ 0.01 } };

    // set csvm parameter to new values
    csvm.set_params(params);

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}
TYPED_TEST(BaseCSVM, set_params_from_named_parameters) {
    using real_type = TypeParam;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    mock_csvm<real_type> csvm{};

    // create parameter class
    const plssvm::parameter<real_type> params{ plssvm::kernel_function_type::polynomial, 4, real_type{ 0.2 }, real_type{ 0.1 }, real_type{ 0.01 } };

    // set csvm parameter to new values
    csvm.set_params(plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
                    plssvm::degree = 4,
                    plssvm::gamma = 0.2,
                    plssvm::coef0 = 0.1,
                    plssvm::cost = 0.01);

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

template <typename T>
class BaseCSVMFit : public BaseCSVM<T>, private util::redirect_output, protected util::temporary_file {};
TYPED_TEST_SUITE(BaseCSVMFit, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(BaseCSVMFit, fit) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the solve_system_of_linear_equations function
    EXPECT_CALL(csvm, solve_system_of_linear_equations).Times(1);

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<real_type, label_type> training_data{ this->filename };

    // call function
    const plssvm::model<real_type, label_type> model = csvm.fit(training_data);
}
TYPED_TEST(BaseCSVMFit, fit_named_parameters) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the solve_system_of_linear_equations function
    EXPECT_CALL(csvm, solve_system_of_linear_equations).Times(1);

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<real_type, label_type> training_data{ this->filename };

    // call function
    const plssvm::model<real_type, label_type> model = csvm.fit(training_data, plssvm::epsilon = 0.1, plssvm::max_iter = 10);
}
TYPED_TEST(BaseCSVMFit, fit_named_parameters_invalid_epsilon) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the solve_system_of_linear_equations function -> since an exception should be triggered, the mocked function should never be called
    EXPECT_CALL(csvm, solve_system_of_linear_equations).Times(0);

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<real_type, label_type> training_data{ this->filename };

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvm.fit(training_data, plssvm::epsilon = 0.0)),
                      plssvm::invalid_parameter_exception,
                      "epsilon must be less than 0.0, but is 0!");
}
TYPED_TEST(BaseCSVMFit, fit_named_parameters_invalid_max_iter) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the solve_system_of_linear_equations function -> since an exception should be triggered, the mocked function should never be called
    EXPECT_CALL(csvm, solve_system_of_linear_equations).Times(0);

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<real_type, label_type> training_data{ this->filename };

    // calling the function with an invalid max_iter should throw
    EXPECT_THROW_WHAT((std::ignore = csvm.fit(training_data, plssvm::max_iter = 0)),
                      plssvm::invalid_parameter_exception,
                      "max_iter must be greater than 0, but is 0!");
}
TYPED_TEST(BaseCSVMFit, fit_no_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the solve_system_of_linear_equations function -> since an exception should be triggered, the mocked function should never be called
    EXPECT_CALL(csvm, solve_system_of_linear_equations).Times(0);

    // create data set without labels
    const plssvm::data_set<real_type, label_type> training_data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // in order to call fit, the provided data set must contain labels
    EXPECT_THROW_WHAT((std::ignore = csvm.fit(training_data)),
                      plssvm::invalid_parameter_exception,
                      "No labels given for training! Maybe the data is only usable for prediction?");
}

template <typename T>
class BaseCSVMPredict : public BaseCSVM<T>, private util::redirect_output {};
TYPED_TEST_SUITE(BaseCSVMPredict, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(BaseCSVMPredict, predict) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the predict_values function
    EXPECT_CALL(csvm, predict_values).Times(1);

    // create data set
    const util::temporary_file data_set_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", data_set_file.filename);
    const plssvm::data_set<real_type, label_type> data_to_predict{ data_set_file.filename };

    // read a previously learned from a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> learned_model{ model_file.filename };

    // call function
    const std::vector<label_type> prediction = csvm.predict(learned_model, data_to_predict);
}
TYPED_TEST(BaseCSVMPredict, predict_num_feature_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    EXPECT_CALL(csvm, predict_values).Times(0);

    // create data set
    const plssvm::data_set<real_type, label_type> data_to_predict{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // read a previously learned from a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> learned_model{ model_file.filename };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvm.predict(learned_model, data_to_predict),
                      plssvm::invalid_parameter_exception,
                      "Number of features per data point (2) must match the number of features per support vector of the provided model (4)!");
}

template <typename T>
class BaseCSVMScore : public BaseCSVM<T>, private util::redirect_output {};
TYPED_TEST_SUITE(BaseCSVMScore, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(BaseCSVMScore, score_model) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the predict_values function
    EXPECT_CALL(csvm, predict_values).Times(1);

    // read a previously learned from a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> learned_model{ model_file.filename };

    // call function
    [[maybe_unused]] const real_type score = csvm.score(learned_model);
}
TYPED_TEST(BaseCSVMScore, score_data_set) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the predict_values function
    EXPECT_CALL(csvm, predict_values).Times(1);

    // create data set
    const util::temporary_file data_set_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", data_set_file.filename);
    const plssvm::data_set<real_type, label_type> data_to_score{ data_set_file.filename };

    // read a previously learned from a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> learned_model{ model_file.filename };

    // call function
    [[maybe_unused]] const real_type score = csvm.score(learned_model, data_to_score);
}

TYPED_TEST(BaseCSVMScore, score_data_set_no_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    EXPECT_CALL(csvm, predict_values).Times(0);

    // create data set
    const plssvm::data_set<real_type, label_type> data_to_score{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // read a previously learned from a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> learned_model{ model_file.filename };

    // in order to call score, the provided data set must contain labels
    EXPECT_THROW_WHAT(std::ignore = csvm.score(learned_model, data_to_score), plssvm::invalid_parameter_exception, "The data set to score must have labels!");
}
TYPED_TEST(BaseCSVMScore, score_data_set_num_features_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    EXPECT_CALL(csvm, predict_values).Times(0);

    // create data set
    const plssvm::data_set<real_type, label_type> data_to_score{
        std::vector<std::vector<real_type>>{ { real_type{ 1.0 }, real_type{ 2.0 } }, { real_type{ 3.0 }, real_type{ 4.0 } } },
        std::vector<label_type>{ util::get_distinct_label<label_type>().first, util::get_distinct_label<label_type>().second }
    };

    // read a previously learned from a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> learned_model{ model_file.filename };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvm.score(learned_model, data_to_score),
                      plssvm::invalid_parameter_exception,
                      "Number of features per data point (2) must match the number of features per support vector of the provided model (4)!");
}
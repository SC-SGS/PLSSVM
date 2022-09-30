/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base functionality independent of the used backend.
 */

#include "mock_csvm.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception, plssvm::file_not_found_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "backends/compare.hpp"  // compare::kernel_function
#include "utility.hpp"           // util::gtest_expect_floating_point_eq, util::google_test::parameter_definition, util::google_test::parameter_definition_to_name,
                                 // util::create_temp_file, EXPECT_THROW_WHAT

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_THAT, EXPECT_CALL

#include <algorithm>   // std::generate
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <memory>      // std::make_shared
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string
#include <vector>      // std::vector

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class BaseCSVM : public ::testing::Test {};
TYPED_TEST_SUITE(BaseCSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether plssvm::csvm<T>::learn() internally calls the correct functions
TYPED_TEST(BaseCSVM, learn) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    EXPECT_CALL(csvm, setup_data_on_device).Times(1);
    EXPECT_CALL(csvm, generate_q).Times(1);
    EXPECT_CALL(csvm, solver_CG).Times(1);

    csvm.learn();
}

// check whether plssvm::csvm<T>::learn() with wrong data correctly fails
TYPED_TEST(BaseCSVM, learn_exceptions) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // functions shouldn't be called, since the exceptions must trigger beforehand
    EXPECT_CALL(csvm, setup_data_on_device).Times(0);
    EXPECT_CALL(csvm, generate_q).Times(0);
    EXPECT_CALL(csvm, solver_CG).Times(0);

    // attempting to call learn() without any labels specified should result in an exception
    csvm.get_value_ptr() = nullptr;
    EXPECT_THROW_WHAT(csvm.learn(), plssvm::exception, "No labels given for training! Maybe the data is only usable for prediction?");

    // attempting to call learn() with different number of labels than data points should result in an exception
    csvm.get_value_ptr() = std::make_shared<const std::vector<typename TypeParam::real_type>>();
    EXPECT_THROW_WHAT(csvm.learn(), plssvm::exception, "Number of labels (0) must match the number of data points (5)!");
}

// check whether plssvm::csvm<T>::accuracy() with an empty points vector returns an accuracy of 0
TYPED_TEST(BaseCSVM, accuracy_empty_points) {
    using real_type = typename TypeParam::real_type;
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // predict should never be called, since the exceptions must trigger beforehand
    EXPECT_CALL(csvm, predict).Times(0);

    // the number of provided points and provided correct labels must match
    const std::vector<std::vector<real_type>> points{};
    const std::vector<real_type> correct_labels{};
    util::gtest_expect_floating_point_eq(real_type{ 0 }, csvm.accuracy(points, correct_labels));
}

// check whether plssvm::csvm<T>::accuracy() without labels or wrong number if features specified correctly fails
TYPED_TEST(BaseCSVM, accuracy_exceptions) {
    using real_type = typename TypeParam::real_type;
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // predict should never be called, since the exceptions must trigger beforehand
    EXPECT_CALL(csvm, predict).Times(0);

    [[maybe_unused]] real_type acc;
    // attempting to call accuracy() with the wrong number of features should result in an exception
    EXPECT_THROW_WHAT(acc = csvm.accuracy({}, -1), plssvm::exception, "Number of features per data point (4) must match the number of features of the predict point (0)!");

    // the number of provided points and provided correct labels must match
    std::vector<std::vector<real_type>> points = { { real_type{ 1 } }, { real_type{ 2 } } };
    std::vector<real_type> correct_labels = { real_type{ -1 } };
    EXPECT_THROW_WHAT(acc = csvm.accuracy(points, correct_labels), plssvm::exception, "Number of data points (2) must match number of correct labels (1)!");

    // the number of features of the prediction points must all be the same
    points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 } } };
    correct_labels = { real_type{ -1 }, real_type{ 1 } };
    EXPECT_THROW_WHAT(acc = csvm.accuracy(points, correct_labels), plssvm::exception, "All points in the prediction point vector must have the same number of features!");

    // the number of features of the prediction points must match the number of features of the data points
    points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 }, real_type{ 4 } } };
    EXPECT_THROW_WHAT(acc = csvm.accuracy(points, correct_labels), plssvm::exception, "Number of features per data point (4) must match the number of features per predict point (2)!");

    // attempting to call accuracy() without any labels specified should result in an exception
    csvm.get_value_ptr() = nullptr;
    EXPECT_THROW_WHAT(acc = csvm.accuracy(), plssvm::exception, "No labels given! Maybe the data is only usable for prediction?");
}

// check whether plssvm::csvm<T>::predict_label() with an empty points vector returns an empty vector
TYPED_TEST(BaseCSVM, predict_label_empty_points) {
    using real_type = typename TypeParam::real_type;
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // predict should never be called, since the exceptions must trigger beforehand
    EXPECT_CALL(csvm, predict).Times(0);

    // the number of provided points and provided correct labels must match
    const std::vector<std::vector<real_type>> points{};
    ASSERT_EQ(std::vector<real_type>{}, csvm.predict_label(points));
}

// check whether plssvm::csvm<T>::predict_label() without labels or wrong number if features specified correctly fails
TYPED_TEST(BaseCSVM, predict_label_exceptions) {
    using real_type = typename TypeParam::real_type;
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // predict should never be called, since the exceptions must trigger beforehand
    EXPECT_CALL(csvm, predict).Times(0);

    [[maybe_unused]] real_type acc;
    // attempting to call predict_label() with the wrong number of features should result in an exception
    std::vector<real_type> point{};
    EXPECT_THROW_WHAT(acc = csvm.predict_label(point), plssvm::exception, "Number of features per data point (4) must match the number of features of the predict point (0)!");

    [[maybe_unused]] std::vector<real_type> acc_vec;
    // the number of features of the prediction points must all be the same
    std::vector<std::vector<real_type>> points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 } } };
    EXPECT_THROW_WHAT(acc_vec = csvm.predict_label(points), plssvm::exception, "All points in the prediction point vector must have the same number of features!");

    // the number of features of the prediction points must match the number of features of the data points
    points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 }, real_type{ 4 } } };
    EXPECT_THROW_WHAT(acc_vec = csvm.predict_label(points), plssvm::exception, "Number of features per data point (4) must match the number of features per predict point (2)!");
}

// check whether plssvm::csvm<T>::predict_label() without labels or wrong number if features specified correctly fails
TYPED_TEST(BaseCSVM, predict_exceptions) {
    using real_type = typename TypeParam::real_type;
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // predict should never be called, since the exceptions must trigger beforehand
    EXPECT_CALL(csvm, predict).Times(0);

    [[maybe_unused]] real_type label;
    // attempting to call predict_label() with the wrong number of features should result in an exception
    const std::vector<real_type> point;
    EXPECT_THROW_WHAT(label = csvm.predict(point), plssvm::exception, "Number of features per data point (4) must match the number of features of the predict point (0)!");
}
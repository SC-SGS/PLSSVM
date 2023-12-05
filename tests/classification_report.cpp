/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the classification report.
 */

#include "plssvm/classification_report.hpp"

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING
#include "utility.hpp"             // util::redirect_output

#include "gtest/gtest.h"  // TEST, TEST_F, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ::testing::Test

#include <cmath>     // std::isnan
#include <iostream>  // std::cout
#include <sstream>   // std::istringstream
#include <string>    // std::string
#include <tuple>     // std::ignore
#include <vector>    // std::vector

//*************************************************************************************************************************************//
//                                                        zero division behavior                                                       //
//*************************************************************************************************************************************//

class ZeroDivisionBehavior : public ::testing::Test, public util::redirect_output<&std::cout> {};

// check whether the plssvm::classification_report::zero_division_behavior -> std::string conversions are correct
TEST_F(ZeroDivisionBehavior, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::classification_report::zero_division_behavior::warn, "warn");
    EXPECT_CONVERSION_TO_STRING(plssvm::classification_report::zero_division_behavior::zero, "0.0");
    EXPECT_CONVERSION_TO_STRING(plssvm::classification_report::zero_division_behavior::one, "1.0");
    EXPECT_CONVERSION_TO_STRING(plssvm::classification_report::zero_division_behavior::nan, "nan");
}
TEST_F(ZeroDivisionBehavior, to_string_unknown) {
    // check conversions to std::string from unknown zero_division_behavior
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::classification_report::zero_division_behavior>(4), "unknown");
}

// check whether the std::string -> plssvm::classification_report::zero_division_behavior conversions are correct
TEST_F(ZeroDivisionBehavior, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("warn", plssvm::classification_report::zero_division_behavior::warn);
    EXPECT_CONVERSION_FROM_STRING("WARN", plssvm::classification_report::zero_division_behavior::warn);
    EXPECT_CONVERSION_FROM_STRING("zero", plssvm::classification_report::zero_division_behavior::zero);
    EXPECT_CONVERSION_FROM_STRING("0.0", plssvm::classification_report::zero_division_behavior::zero);
    EXPECT_CONVERSION_FROM_STRING("one", plssvm::classification_report::zero_division_behavior::one);
    EXPECT_CONVERSION_FROM_STRING("1.0", plssvm::classification_report::zero_division_behavior::one);
    EXPECT_CONVERSION_FROM_STRING("nan", plssvm::classification_report::zero_division_behavior::nan);
    EXPECT_CONVERSION_FROM_STRING("NaN", plssvm::classification_report::zero_division_behavior::nan);
}
TEST_F(ZeroDivisionBehavior, from_string_unknown) {
    // foo isn't a valid zero_division_behavior
    std::istringstream input{ "foo" };
    plssvm::classification_report::zero_division_behavior zero_div{};
    input >> zero_div;
    EXPECT_TRUE(input.fail());
}

TEST_F(ZeroDivisionBehavior, sanitize_nan_warn) {
    // sanitize NaN using warn
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 0.0, plssvm::classification_report::zero_division_behavior::warn, "Foo"), 0.0);
    EXPECT_EQ(this->get_capture(), "Foo is ill-defined and is set to 0.0 in labels with no predicted samples. "
                                   "Use 'plssvm::classification_report::zero_division' parameter to control this behavior.\n");
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 1.0, plssvm::classification_report::zero_division_behavior::warn, "Foo"), 42.0);
}
TEST_F(ZeroDivisionBehavior, sanitize_nan_zero) {
    // sanitize NaN using zero
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 0.0, plssvm::classification_report::zero_division_behavior::zero, "Foo"), 0.0);
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 1.0, plssvm::classification_report::zero_division_behavior::zero, "Foo"), 42.0);
}
TEST_F(ZeroDivisionBehavior, sanitize_nan_one) {
    // sanitize NaN using one
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 0.0, plssvm::classification_report::zero_division_behavior::one, "Foo"), 1.0);
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 1.0, plssvm::classification_report::zero_division_behavior::one, "Foo"), 42.0);
}
TEST_F(ZeroDivisionBehavior, sanitize_nan_nan) {
    // sanitize NaN using nan
#if !defined(NDEBUG) || defined(_MSC_VER)
    // ATTENTION: MSVC doesn't optimize out the NaN check even if fast math is used
    EXPECT_TRUE(std::isnan(plssvm::detail::sanitize_nan(42.0, 0.0, plssvm::classification_report::zero_division_behavior::nan, "Foo")));
#else
    // ATTENTION: std::isnan will ALWAYS return false due to -ffast-math being enabled in release mode (in GCC and clang)
    EXPECT_FALSE(std::isnan(plssvm::detail::sanitize_nan(42.0, 0.0, plssvm::classification_report::zero_division_behavior::nan, "Foo")));
#endif
    EXPECT_EQ(plssvm::detail::sanitize_nan(42.0, 1.0, plssvm::classification_report::zero_division_behavior::nan, "Foo"), 42.0);
}

//*************************************************************************************************************************************//
//                                                               metrics                                                               //
//*************************************************************************************************************************************//

TEST(ClassificationReportMetrics, construct_metric) {
    // construct a metric object
    const plssvm::classification_report::metric m{ 10, 42, 3, 0.1, 0.2, 0.3, 42 };

    // check if values are set correctly
    EXPECT_EQ(m.TP, 10);
    EXPECT_EQ(m.FP, 42);
    EXPECT_EQ(m.FN, 3);
    EXPECT_EQ(m.precision, 0.1);
    EXPECT_EQ(m.recall, 0.2);
    EXPECT_EQ(m.f1, 0.3);
    EXPECT_EQ(m.support, 42);
}
TEST(ClassificationReportMetrics, output_metric) {
    // construct a metric object
    EXPECT_CONVERSION_TO_STRING((plssvm::classification_report::metric{ 10, 42, 3, 0.1, 0.2, 0.3, 42 }),
                                "TP: 10\n"
                                "FP: 42\n"
                                "FN: 3\n"
                                "precision: 0.1\n"
                                "recall:    0.2\n"
                                "f1-score:  0.3\n"
                                "support:   42");
}

TEST(ClassificationReportMetrics, construct_accuracy_metric) {
    // construct a accuracy metric object
    const plssvm::classification_report::accuracy_metric am{ 0.5, 50, 100 };

    // check if values are set correctly
    EXPECT_EQ(am.achieved_accuracy, 0.5);
    EXPECT_EQ(am.num_correct, 50);
    EXPECT_EQ(am.num_total, 100);
}
TEST(ClassificationReportMetrics, output_accuracy_metric) {
    // construct a accuracy metric object
    EXPECT_CONVERSION_TO_STRING((plssvm::classification_report::accuracy_metric{ 0.5, 50, 100 }), "Accuracy = 50.00% (50/100)");
}

class ClassificationReport : public ::testing::Test {
  protected:
    /**
     * @brief Return the correct labels to calculate the classification report with.
     * @return the correct labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<int> &get_correct_label() const noexcept {
        return correct_label_;
    }
    /**
     * @brief Return the predicted labels to calculate the classification report with.
     * @return the predicted labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<int> &get_predicted_label() const noexcept {
        return predicted_label_;
    }
    /**
     * @brief Return the confusion matrix for @p correct_label land @p predict_label.
     * @return the confusion matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::aos_matrix<unsigned long long> &get_confusion_matrix() const noexcept { return confusion_matrix_; }

  private:
    // clang-format off
    /// The correct class labels.
    std::vector<int> correct_label_ = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    /// The predicted class labels.
    std::vector<int> predicted_label_ = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1,
                                          1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 };
    /// The confusion matrix resulting from correct_label and predicted_label.
    plssvm::aos_matrix<unsigned long long> confusion_matrix_{ { { 9, 1, 5 },
                                                                { 6, 7, 4 },
                                                                { 3, 2, 8 } }
    };
    // clang-format on
};

TEST_F(ClassificationReport, construct) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check if values are set correctly

    EXPECT_EQ(report.confusion_matrix(), this->get_confusion_matrix());

    plssvm::classification_report::metric m = report.metric_for_class(0);
    EXPECT_EQ(m.TP, 9);
    EXPECT_EQ(m.FP, 9);
    EXPECT_EQ(m.FN, 6);
    EXPECT_EQ(m.precision, 0.5);
    EXPECT_EQ(m.recall, 0.6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.54545454545, 1e6);
    EXPECT_EQ(m.support, 15);

    m = report.metric_for_class(1);
    EXPECT_EQ(m.TP, 7);
    EXPECT_EQ(m.FP, 3);
    EXPECT_EQ(m.FN, 10);
    EXPECT_EQ(m.precision, 0.7);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.recall, 0.41176470588, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.51851851851, 1e6);
    EXPECT_EQ(m.support, 17);

    m = report.metric_for_class(2);
    EXPECT_EQ(m.TP, 8);
    EXPECT_EQ(m.FP, 9);
    EXPECT_EQ(m.FN, 5);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.precision, 0.47058823529, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.recall, 0.61538461538, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.53333333332, 1e6);
    EXPECT_EQ(m.support, 13);

    EXPECT_FLOATING_POINT_NEAR_EPS(report.accuracy().achieved_accuracy, 0.53333333333, 1e6);
    EXPECT_EQ(report.accuracy().num_correct, 24);
    EXPECT_EQ(report.accuracy().num_total, 45);
}
TEST_F(ClassificationReport, construct_target_names) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label(), plssvm::classification_report::target_names = std::vector<std::string>{ "Foo", "Bar", "Baz" } };

    // check if values are set correctly
    EXPECT_EQ(report.confusion_matrix(), this->get_confusion_matrix());

    plssvm::classification_report::metric m = report.metric_for_class("Foo");
    EXPECT_EQ(m.TP, 9);
    EXPECT_EQ(m.FP, 9);
    EXPECT_EQ(m.FN, 6);
    EXPECT_EQ(m.precision, 0.5);
    EXPECT_EQ(m.recall, 0.6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.54545454545, 1e6);
    EXPECT_EQ(m.support, 15);

    m = report.metric_for_class("Bar");
    EXPECT_EQ(m.TP, 7);
    EXPECT_EQ(m.FP, 3);
    EXPECT_EQ(m.FN, 10);
    EXPECT_EQ(m.precision, 0.7);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.recall, 0.41176470588, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.51851851851, 1e6);
    EXPECT_EQ(m.support, 17);

    m = report.metric_for_class("Baz");
    EXPECT_EQ(m.TP, 8);
    EXPECT_EQ(m.FP, 9);
    EXPECT_EQ(m.FN, 5);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.precision, 0.47058823529, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.recall, 0.61538461538, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.53333333332, 1e6);
    EXPECT_EQ(m.support, 13);

    EXPECT_FLOATING_POINT_NEAR_EPS(report.accuracy().achieved_accuracy, 0.53333333333, 1e6);
    EXPECT_EQ(report.accuracy().num_correct, 24);
    EXPECT_EQ(report.accuracy().num_total, 45);
}
TEST_F(ClassificationReport, construct_target_names_size_mismatch) {
    // too few new target names
    EXPECT_THROW_WHAT((plssvm::classification_report{ this->get_correct_label(), this->get_predicted_label(), plssvm::classification_report::target_names = std::vector<std::string>{ "Foo", "Bar" } }),
                      plssvm::exception,
                      "Provided 2 target names, but found 3 distinct labels!");
    // too many new target names
    EXPECT_THROW_WHAT((plssvm::classification_report{ this->get_correct_label(), this->get_predicted_label(), plssvm::classification_report::target_names = std::vector<std::string>{ "Foo", "Bar", "Baz", "Bat" } }),
                      plssvm::exception,
                      "Provided 4 target names, but found 3 distinct labels!");
}

TEST_F(ClassificationReport, construct_zero_division_behavior) {
    // construct a classification report
    const plssvm::classification_report report{ std::vector<int>{ 0, 0, 0 }, std::vector<int>{ 1, 1, 1 }, plssvm::classification_report::zero_division = plssvm::classification_report::zero_division_behavior::one };

    // check if values are set correctly
    const plssvm::aos_matrix<unsigned long long> correct_confusion_matrix{ { { 0, 3 }, { 0, 0 } } };
    EXPECT_EQ(report.confusion_matrix(), correct_confusion_matrix);

    plssvm::classification_report::metric m = report.metric_for_class(0);
    EXPECT_EQ(m.TP, 0);
    EXPECT_EQ(m.FP, 0);
    EXPECT_EQ(m.FN, 3);
    EXPECT_EQ(m.precision, 1.0);
    EXPECT_EQ(m.recall, 0.0);
    EXPECT_EQ(m.f1, 0.0);
    EXPECT_EQ(m.support, 3);

    m = report.metric_for_class(1);
    EXPECT_EQ(m.TP, 0);
    EXPECT_EQ(m.FP, 3);
    EXPECT_EQ(m.FN, 0);
    EXPECT_EQ(m.precision, 0.0);
    EXPECT_EQ(m.recall, 1.0);
    EXPECT_EQ(m.f1, 0.0);
    EXPECT_EQ(m.support, 0);

    EXPECT_EQ(report.accuracy().achieved_accuracy, 0.0);
    EXPECT_EQ(report.accuracy().num_correct, 0);
    EXPECT_EQ(report.accuracy().num_total, 3);
}

TEST_F(ClassificationReport, construct_digits_negative) {
    // too few new target names
    EXPECT_THROW_WHAT((plssvm::classification_report{ this->get_correct_label(), this->get_predicted_label(), plssvm::classification_report::digits = -1 }),
                      plssvm::exception,
                      "Invalid number of output digits provided! Number of digits must be greater than zero but is -1!");
}

TEST_F(ClassificationReport, construct_empty_correct_label) {
    // the correct labels vector must not be empty
    EXPECT_THROW_WHAT((plssvm::classification_report{ std::vector<int>{}, this->get_predicted_label() }),
                      plssvm::exception,
                      "The correct labels list must not be empty!");
}
TEST_F(ClassificationReport, construct_empty_predicted_label) {
    // the predicted labels vector must not be empty
    EXPECT_THROW_WHAT((plssvm::classification_report{ this->get_correct_label(), std::vector<int>{} }),
                      plssvm::exception,
                      "The predicted labels list must not be empty!");
}
TEST_F(ClassificationReport, construct_label_size_mismatch) {
    // constructing a classification report with different number of correct and predicted labels must throw
    EXPECT_THROW_WHAT((plssvm::classification_report{ std::vector<int>{ 0, 0, 0 }, std::vector<int>{ 0, 0 } }),
                      plssvm::exception,
                      "The number of correct labels (3) and predicted labels (2) must be the same!");
}

TEST_F(ClassificationReport, confusion_matrix) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check if the confusion matrix is correct
    EXPECT_EQ(report.confusion_matrix(), this->get_confusion_matrix());
}
TEST_F(ClassificationReport, accuracy_metrics) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check if the accuracy metrics are set correctly
    EXPECT_FLOATING_POINT_NEAR_EPS(report.accuracy().achieved_accuracy, 0.53333333333, 1e6);
    EXPECT_EQ(report.accuracy().num_correct, 24);
    EXPECT_EQ(report.accuracy().num_total, 45);
}
TEST_F(ClassificationReport, metric_for_class) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check the precision, recall, f1 score, and support metrics for all labels
    // label 0
    plssvm::classification_report::metric m = report.metric_for_class(0);
    EXPECT_EQ(m.TP, 9);
    EXPECT_EQ(m.FP, 9);
    EXPECT_EQ(m.FN, 6);
    EXPECT_EQ(m.precision, 0.5);
    EXPECT_EQ(m.recall, 0.6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.54545454545, 1e6);
    EXPECT_EQ(m.support, 15);

    // label 1
    m = report.metric_for_class(1);
    EXPECT_EQ(m.TP, 7);
    EXPECT_EQ(m.FP, 3);
    EXPECT_EQ(m.FN, 10);
    EXPECT_EQ(m.precision, 0.7);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.recall, 0.41176470588, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.51851851851, 1e6);
    EXPECT_EQ(m.support, 17);

    // label 2
    m = report.metric_for_class(2);
    EXPECT_EQ(m.TP, 8);
    EXPECT_EQ(m.FP, 9);
    EXPECT_EQ(m.FN, 5);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.precision, 0.47058823529, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.recall, 0.61538461538, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.f1, 0.53333333332, 1e6);
    EXPECT_EQ(m.support, 13);

    EXPECT_EQ(report.metric_for_class(0).support + report.metric_for_class(1).support + report.metric_for_class(2).support, this->get_correct_label().size());
}
TEST_F(ClassificationReport, metric_for_invalid_class) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label() };

    // try checking for an illegal class
    EXPECT_THROW_WHAT(std::ignore = report.metric_for_class(-1), plssvm::exception, "Couldn't find the label \"-1\"!");
    EXPECT_THROW_WHAT(std::ignore = report.metric_for_class("foo"), plssvm::exception, "Couldn't find the label \"foo\"!");
}

TEST_F(ClassificationReport, classification_report) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check output
    const std::string correct_output =
        "              precision    recall  f1-score   support\n"
        "\n"
        "           0       0.50      0.60      0.55        15\n"
        "           1       0.70      0.41      0.52        17\n"
        "           2       0.47      0.62      0.53        13\n"
        "\n"
        "    accuracy                           0.53        45\n"
        "   macro avg       0.56      0.54      0.53        45\n"
        "weighted avg       0.57      0.53      0.53        45\n\n"
        "Accuracy = 53.33% (24/45)\n";
    EXPECT_CONVERSION_TO_STRING(report, correct_output);
}
TEST_F(ClassificationReport, classification_report_target_names) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label(), plssvm::classification_report::target_names = std::vector<std::string>{ "cat", "dog", "African elephant" } };

    // check output
    const std::string correct_output =
        "                  precision    recall  f1-score   support\n"
        "\n"
        "             cat       0.50      0.60      0.55        15\n"
        "             dog       0.70      0.41      0.52        17\n"
        "African elephant       0.47      0.62      0.53        13\n"
        "\n"
        "        accuracy                           0.53        45\n"
        "       macro avg       0.56      0.54      0.53        45\n"
        "    weighted avg       0.57      0.53      0.53        45\n\n"
        "Accuracy = 53.33% (24/45)\n";
    EXPECT_CONVERSION_TO_STRING(report, correct_output);
}
TEST_F(ClassificationReport, classification_report_digits) {
    // construct a classification report
    const plssvm::classification_report report{ this->get_correct_label(), this->get_predicted_label(), plssvm::classification_report::digits = 3 };

    // check output
    const std::string correct_output =
        "               precision     recall   f1-score   support\n"
        "\n"
        "           0       0.500      0.600      0.545        15\n"
        "           1       0.700      0.412      0.519        17\n"
        "           2       0.471      0.615      0.533        13\n"
        "\n"
        "    accuracy                             0.533        45\n"
        "   macro avg       0.557      0.542      0.532        45\n"
        "weighted avg       0.567      0.533      0.532        45\n\n"
        "Accuracy = 53.33% (24/45)\n";
    EXPECT_CONVERSION_TO_STRING(report, correct_output);
}
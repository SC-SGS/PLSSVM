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

#include "plssvm/backends/SYCL/detail/constants.hpp"

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_EQ

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, ::testing::Test

#include <cstddef>  // std::size_t
#include <sstream>  // std::ostringstream
#include <string>   // std::string
#include <vector>   // std::vector

class ClassificationReport : public ::testing::Test {
  protected:
    const std::vector<int> correct_label = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 };
    const std::vector<int> predicted_label = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 };
};
class ClassificationReportDeathTest : public ClassificationReport {};

TEST_F(ClassificationReport, construct_metric) {
    // construct a metric object
    const plssvm::classification_report::metric m{ 0.1, 0.2, 0.3, 42 };

    // check if values are set correctly
    EXPECT_EQ(m.precision, 0.1);
    EXPECT_EQ(m.recall, 0.2);
    EXPECT_EQ(m.f1, 0.3);
    EXPECT_EQ(m.support, 42);
}
TEST_F(ClassificationReport, construct_accuracy_metric) {
    // construct a accuracy metric object
    const plssvm::classification_report::accuracy_metric am{ 0.5, 50, 100 };

    // check if values are set correctly
    EXPECT_EQ(am.achieved_accuracy, 0.5);
    EXPECT_EQ(am.num_correct, 50);
    EXPECT_EQ(am.num_total, 100);
    EXPECT_EQ(am.achieved_accuracy, static_cast<double>(am.num_correct) / static_cast<double>(am.num_total));
}

TEST_F(ClassificationReport, construct_all) {
    // construct a classification report
    const plssvm::classification_report report{ this->correct_label, this->predicted_label };

    // check if values are set correctly
    const std::vector<std::vector<std::size_t>> correct_confusion_matrix{
        { 9, 1, 5 },
        { 6, 7, 4 },
        { 3, 2, 8 }
    };
    EXPECT_EQ(report.confusion_matrix(), correct_confusion_matrix);
    EXPECT_EQ(report.accuracy().achieved_accuracy, 24.0 / 45.0);
    EXPECT_EQ(report.accuracy().num_correct, 24);
    EXPECT_EQ(report.accuracy().num_total, 45);
}

TEST_F(ClassificationReport, construct_label_size_mismatch) {
    // constructing a classification report with different number of correct and predicted labels must throw
    EXPECT_THROW_WHAT((plssvm::classification_report{ std::vector<int>{ 0, 0, 0 }, std::vector<int>{ 0, 0 } }),
                      plssvm::exception,
                      "The number of correct labels (3) and predicted labels (2) must be the same!");
}
TEST_F(ClassificationReportDeathTest, construct_empty_correct_label) {
    // the correct labels vector must not be empty
    EXPECT_DEATH((plssvm::classification_report{ std::vector<int>{}, this->predicted_label }), "The correct labels list may not be empty!");
}
TEST_F(ClassificationReportDeathTest, construct_empty_predicted_label) {
    // the predicted labels vector must not be empty
    EXPECT_DEATH((plssvm::classification_report{ this->correct_label, std::vector<int>{} }), "The predicted labels list may not be empty!");
}

TEST_F(ClassificationReport, confusion_matrix) {
    // construct a classification report
    const plssvm::classification_report report{ this->correct_label, this->predicted_label };

    // check if the confusion matrix is correct
    const std::vector<std::vector<std::size_t>> correct_confusion_matrix{
        { 9, 1, 5 },
        { 6, 7, 4 },
        { 3, 2, 8 }
    };
    EXPECT_EQ(report.confusion_matrix(), correct_confusion_matrix);
}
TEST_F(ClassificationReport, accuracy_metrics) {
    // construct a classification report
    const plssvm::classification_report report{ this->correct_label, this->predicted_label };

    // check if the accuracy metrics are set correctly
    EXPECT_EQ(report.accuracy().achieved_accuracy, 24.0 / 45.0);
    EXPECT_EQ(report.accuracy().num_correct, 24);
    EXPECT_EQ(report.accuracy().num_total, 45);
}
TEST_F(ClassificationReport, metric_for_class) {
    // construct a classification report
    const plssvm::classification_report report{ this->correct_label, this->predicted_label };

    // check the precision, recall, f1 score, and support metrics for all labels
    // label 0
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(0).precision, 0.5);
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(0).recall, 0.6);
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(0).f1, 0.5454545454545454);
    EXPECT_EQ(report.metric_for_class(0).support, 15);
    // label 1
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(1).precision, 0.7);
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(1).recall, 0.4117647058823529);
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(1).f1, 0.5185185185185185);
    EXPECT_EQ(report.metric_for_class(1).support, 17);
    // label 2
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(2).precision, 0.47058823529411764);
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(2).recall, 0.6153846153846154);
    EXPECT_FLOATING_POINT_EQ(report.metric_for_class(2).f1, 0.5333333333333333);
    EXPECT_EQ(report.metric_for_class(2).support, 13);

    EXPECT_EQ(report.metric_for_class(0).support + report.metric_for_class(1).support + report.metric_for_class(2).support, this->correct_label.size());
}

TEST_F(ClassificationReport, output_accuracy_metric) {
    // construct a classification report
    const plssvm::classification_report report{ this->correct_label, this->predicted_label };

    // output accuracy metric
    std::ostringstream out;
    out << report.accuracy();

    // check output
    EXPECT_EQ(out.str(), "Accuracy = 53.33% (24/45)");
}
TEST_F(ClassificationReport, classification_report) {
    // construct a classification report
    const plssvm::classification_report report{ this->correct_label, this->predicted_label };

    // output accuracy metric
    std::ostringstream out;
    out << report;

    // check output
    const std::string correct_output =
        "               precision   recall     f1   support\n"
        "           2        0.47     0.62   0.53        13\n"
        "           0        0.50     0.60   0.55        15\n"
        "           1        0.70     0.41   0.52        17\n"
        "\n"
        "    accuracy                        0.53        45\n"
        "   macro avg        0.56     0.54   0.53        45\n"
        "weighted avg        0.57     0.53   0.53        45\n";
    EXPECT_EQ(out.str(), correct_output);
}
/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the regression report.
 */

#include "plssvm/regression_report.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::regression_report_exception

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_CONVERSION_TO_STRING, EXPECT_FLOATING_POINT_NEAR, EXPECT_FLOATING_POINT_NEAR_EPS
#include "tests/utility.hpp"             // util::redirect_output

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::ContainsRegex
#include "gtest/gtest.h"  // TEST, TEST_F, ::testing::Test

#include <iostream>  // std::cout
#include <string>    // std::string
#include <vector>    // std::vector

//*************************************************************************************************************************************//
//                                                               metrics                                                               //
//*************************************************************************************************************************************//

TEST(RegressionReportMetrics, ConstructMetric) {
    // construct a metric object
    const plssvm::regression_report::metric m{ 0.1, 0.2, 0.3, 0.4, 0.5 };

    // check if values are set correctly
    EXPECT_FLOATING_POINT_NEAR(m.explained_variance_score, 0.1);
    EXPECT_FLOATING_POINT_NEAR(m.mean_absolute_error, 0.2);
    EXPECT_FLOATING_POINT_NEAR(m.mean_squared_error, 0.3);
    EXPECT_FLOATING_POINT_NEAR(m.r2_score, 0.4);
    EXPECT_FLOATING_POINT_NEAR(m.squared_correlation_coefficient, 0.5);
}

TEST(RegressionReportMetrics, OutputMetric) {
    // construct a metric object
    EXPECT_CONVERSION_TO_STRING((plssvm::regression_report::metric{ 0.1, 0.2, 0.3, 0.4, 0.5 }),
                                "Explained variance score:        0.1\n"
                                "Mean absolute error:             0.2\n"
                                "Mean squared error:              0.3\n"
                                "R^2 score:                       0.4\n"
                                "Squared correlation coefficient: 0.5");
}

class RegressionReport : public ::testing::Test,
                         public util::redirect_output<> {
  protected:
    /**
     * @brief Return the correct labels to calculate the regression report with.
     * @return the correct labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<plssvm::real_type> &get_correct_label() const noexcept {
        return correct_label_;
    }

    /**
     * @brief Return the predicted labels to calculate the regression report with.
     * @return the predicted labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<plssvm::real_type> &get_predicted_label() const noexcept {
        return predicted_label_;
    }

  private:
    /// The correct class labels.
    std::vector<plssvm::real_type> correct_label_ = { 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 };
    /// The predicted class labels.
    std::vector<plssvm::real_type> predicted_label_ = { 0.1, 0.4, 0.7, 0.8, 1.1, 1.2, 1.5, 1.6, 1.7, 2.0 };
};

TEST_F(RegressionReport, Construct) {
    // construct a regression report
    const plssvm::regression_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check if values are set correctly
    const plssvm::regression_report::metric m = report.loss();
    EXPECT_FLOATING_POINT_NEAR_EPS(m.explained_variance_score, 0.9851515151515151, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.mean_absolute_error, 0.05, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.mean_squared_error, 0.005, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.r2_score, 0.9848484848484849, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.squared_correlation_coefficient, 0.9852899678673179, 1e6);
}

TEST_F(RegressionReport, ConstructPerfectPrediction) {
    // construct a regression report
    const plssvm::regression_report report{ this->get_correct_label(), this->get_correct_label() };

    // check if values are set correctly
    const plssvm::regression_report::metric m = report.loss();
    EXPECT_FLOATING_POINT_NEAR(m.explained_variance_score, 1.0);
    EXPECT_FLOATING_POINT_NEAR(m.mean_absolute_error, 0.0);
    EXPECT_FLOATING_POINT_NEAR(m.mean_squared_error, 0.0);
    EXPECT_FLOATING_POINT_NEAR(m.r2_score, 1.0);
    EXPECT_FLOATING_POINT_NEAR(m.squared_correlation_coefficient, 1.0);
}

TEST_F(RegressionReport, ConstructForceFinite) {
    // construct a regression report
    const plssvm::regression_report report{ this->get_correct_label(), this->get_predicted_label(), plssvm::regression_report::force_finite = true };

    // check if values are set correctly
    const plssvm::regression_report::metric m = report.loss();
    EXPECT_FLOATING_POINT_NEAR_EPS(m.explained_variance_score, 0.9851515151515151, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.mean_absolute_error, 0.05, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.mean_squared_error, 0.005, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.r2_score, 0.9848484848484849, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.squared_correlation_coefficient, 0.9852899678673179, 1e6);
}

TEST_F(RegressionReport, ConstructForceFinitePerfectPrediction) {
    // construct a regression report
    const plssvm::regression_report report{ this->get_correct_label(), this->get_correct_label(), plssvm::regression_report::force_finite = true };

    // check if values are set correctly
    const plssvm::regression_report::metric m = report.loss();
    EXPECT_FLOATING_POINT_NEAR(m.explained_variance_score, 1.0);
    EXPECT_FLOATING_POINT_NEAR(m.mean_absolute_error, 0.0);
    EXPECT_FLOATING_POINT_NEAR(m.mean_squared_error, 0.0);
    EXPECT_FLOATING_POINT_NEAR(m.r2_score, 1.0);
    EXPECT_FLOATING_POINT_NEAR(m.squared_correlation_coefficient, 1.0);
}

TEST_F(RegressionReport, ConstructEmptyCorrectLabel) {
    // the correct labels vector must not be empty
    EXPECT_THROW_WHAT((plssvm::regression_report{ std::vector<plssvm::real_type>{}, this->get_predicted_label() }),
                      plssvm::regression_report_exception,
                      "The correct labels list must not be empty!");
}

TEST_F(RegressionReport, ConstructEmptyPredictedLabel) {
    // the predicted labels vector must not be empty
    EXPECT_THROW_WHAT((plssvm::regression_report{ this->get_correct_label(), std::vector<plssvm::real_type>{} }),
                      plssvm::regression_report_exception,
                      "The predicted labels list must not be empty!");
}

TEST_F(RegressionReport, ConstructLabelSizeMismatch) {
    // constructing a regression report with different number of correct and predicted labels must throw
    EXPECT_THROW_WHAT((plssvm::regression_report{ std::vector<plssvm::real_type>{ 0, 0, 0 }, std::vector<plssvm::real_type>{ 0, 0 } }),
                      plssvm::regression_report_exception,
                      "The number of correct labels (3) and predicted labels (2) must be the same!");
}

TEST_F(RegressionReport, Loss) {
    // construct a regression report
    const plssvm::regression_report report{ this->get_correct_label(), this->get_predicted_label() };

    // check loss values
    const plssvm::regression_report::metric m = report.loss();
    EXPECT_FLOATING_POINT_NEAR_EPS(m.explained_variance_score, 0.9851515151515151, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.mean_absolute_error, 0.05, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.mean_squared_error, 0.005, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.r2_score, 0.9848484848484849, 1e6);
    EXPECT_FLOATING_POINT_NEAR_EPS(m.squared_correlation_coefficient, 0.9852899678673179, 1e6);
}

TEST_F(RegressionReport, RegressionReport) {
    // construct a regression report
    const plssvm::regression_report report{ this->get_correct_label(), this->get_predicted_label() };
    std::cout << report;

    // check output
    const std::string correct_output =
        "Explained variance score:        .*\n"
        "Mean absolute error:             .*\n"
        "Mean squared error:              .*\n"
        "R\\^2 score:                       .*\n"
        "Squared correlation coefficient: .*";
    EXPECT_THAT(this->get_capture(), ::testing::ContainsRegex(correct_output));
}

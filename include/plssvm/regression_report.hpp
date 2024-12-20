/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a regression report returned in the `plssvm::csvr::score` functions.
 */

#ifndef PLSSVM_REGRESSION_REPORT_HPP_
#define PLSSVM_REGRESSION_REPORT_HPP_
#pragma once

#include "plssvm/detail/igor_utility.hpp"    // plssvm::detail::{has_only_named_args_v, get_value_from_named_parameter}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::regression_report_exception

#include "fmt/base.h"     // fmt::formatter
#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // fmt::ostream_formatter
#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT, igor::parser, igor::has_unnamed_arguments, igor::has_other_than

#include <algorithm>  // std::clamp
#include <cmath>      // std::abs
#include <cstddef>    // std::size_t
#include <iosfwd>     // std::ostream
#include <vector>     // std::vector

namespace plssvm {

/**
 * @brief Class calculating a regression report (e.g., MSE or R^2 score).
 */
class regression_report {
  public:
    /// @cond Doxygen_suppress
    // create named arguments
    /// Create a named argument to control whether the R^2 score values should be clipped to 1.0 and 0.0 (and cannot become infinite).
    static IGOR_MAKE_NAMED_ARGUMENT(force_finite);

    /// @endcond

    /**
     * @brief Struct encapsulating the different metrics.
     */
    struct metric {
        /// The explained variance score.
        double explained_variance_score{};
        /// The mean absolute error.
        double mean_absolute_error{};
        /// The mean squared error.
        double mean_squared_error{};
        /// The R^2 score.
        double r2_score{};
        /// The squared correlation coefficient.
        double squared_correlation_coefficient{};
    };

    /**
     * @brief Calculates the regression scores accuracy.
     * @tparam label_type the type of the labels
     * @tparam Args the types of the named arguments
     * @param[in] correct_label the list of correct labels
     * @param[in] predicted_label the list of predicted labels
     * @param[in] named_args the potential name arguments (force_finite)
     * @throws plssvm::regression_report_exception if the @p correct_label or @p predicted_label are empty
     * @throws plssvm::regression_report_exception if the @p correct_label and @p predicted_label sizes mismatch
     */
    template <typename label_type, typename... Args>
    regression_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label, Args &&...named_args);

    /**
     * @brief Return the regression loss.
     * @return the regression loss (`[[nodiscard]]`)
     */
    [[nodiscard]] metric loss() const noexcept { return regression_loss_; }

  private:
    /// The regression loss metrics.
    metric regression_loss_{};
};

template <typename label_type, typename... Args>
regression_report::regression_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label, Args &&...named_args) {
    // sanity check for input correct sizes
    if (correct_label.empty()) {
        throw regression_report_exception{ "The correct labels list must not be empty!" };
    }
    if (predicted_label.empty()) {
        throw regression_report_exception{ "The predicted labels list must not be empty!" };
    }
    if (correct_label.size() != predicted_label.size()) {
        throw regression_report_exception{ fmt::format("The number of correct labels ({}) and predicted labels ({}) must be the same!", correct_label.size(), predicted_label.size()) };
    }

    igor::parser parser{ std::forward<Args>(named_args)... };

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(plssvm::regression_report::force_finite), "An illegal named parameter has been passed!");

    bool force_finite_value{ true };
    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(plssvm::regression_report::force_finite)) {
        // get the value of the provided named parameter
        force_finite_value = detail::get_value_from_named_parameter<decltype(force_finite_value)>(parser, plssvm::regression_report::force_finite);
    }

    // calculate the explained variance score
    {
        double mean_correct{ 0.0 };
        double mean_correct_minus_predicted{ 0.0 };
#pragma omp parallel for default(none) shared(correct_label, predicted_label) reduction(+ : mean_correct, mean_correct_minus_predicted)
        for (std::size_t i = 0; i < correct_label.size(); ++i) {
            mean_correct += static_cast<double>(correct_label[i]);
            mean_correct_minus_predicted += static_cast<double>(correct_label[i] - predicted_label[i]);
        }
        mean_correct /= static_cast<double>(correct_label.size());
        mean_correct_minus_predicted /= static_cast<double>(correct_label.size());

        double variance_correct{ 0.0 };
        double variance_correct_minus_predicted{ 0.0 };
#pragma omp parallel for default(none) shared(correct_label, predicted_label) firstprivate(mean_correct, mean_correct_minus_predicted) reduction(+ : variance_correct, variance_correct_minus_predicted)
        for (std::size_t i = 0; i < correct_label.size(); ++i) {
            variance_correct += static_cast<double>(correct_label[i] - mean_correct) * static_cast<double>(correct_label[i] - mean_correct);
            variance_correct_minus_predicted += static_cast<double>(correct_label[i] - predicted_label[i] - mean_correct_minus_predicted) * static_cast<double>(correct_label[i] - predicted_label[i] - mean_correct_minus_predicted);
        }
        variance_correct /= static_cast<double>(correct_label.size());
        variance_correct_minus_predicted /= static_cast<double>(correct_label.size());
        regression_loss_.explained_variance_score = 1.0 - variance_correct_minus_predicted / variance_correct;
    }

    // calculate the mean absolute error
    {
        double error{ 0.0 };
#pragma omp parallel for default(none) shared(correct_label, predicted_label) reduction(+ : error)
        for (std::size_t i = 0; i < correct_label.size(); ++i) {
            error += std::abs(static_cast<double>(correct_label[i] - predicted_label[i]));
        }
        regression_loss_.mean_absolute_error = error / static_cast<double>(correct_label.size());
    }

    // calculate the mean squared error
    {
        double error{ 0.0 };
#pragma omp parallel for default(none) shared(correct_label, predicted_label) reduction(+ : error)
        for (std::size_t i = 0; i < correct_label.size(); ++i) {
            error += static_cast<double>(correct_label[i] - predicted_label[i]) * static_cast<double>(correct_label[i] - predicted_label[i]);
        }
        regression_loss_.mean_squared_error = error / static_cast<double>(correct_label.size());
    }

    // calculate the R^2 score
    {
        double mean_correct{ 0.0 };
        for (std::size_t i = 0; i < correct_label.size(); ++i) {
            mean_correct += static_cast<double>(correct_label[i]);
        }
        mean_correct /= static_cast<double>(correct_label.size());

        double ss_res{ 0.0 };
        double ss_tot{ 0.0 };
#pragma omp parallel for default(none) shared(correct_label, predicted_label) firstprivate(mean_correct) reduction(+ : ss_res, ss_tot)
        for (std::size_t i = 0; i < correct_label.size(); ++i) {
            ss_res += static_cast<double>(correct_label[i] - predicted_label[i]) * static_cast<double>(correct_label[i] - predicted_label[i]);
            ss_tot += static_cast<double>(correct_label[i] - mean_correct) * static_cast<double>(correct_label[i] - mean_correct);
        }
        regression_loss_.r2_score = 1.0 - ss_res / ss_tot;

        if (force_finite_value) {
#if !defined(PLSSVM_USE_FAST_MATH)
            // R^2 score may not be finite
            if (std::isnan(regression_loss_.r2_score)) {
                // NaN means perfect prediction and is mapped to 1.0
                regression_loss_.r2_score = 1.0;
            } else {
                if (regression_loss_.r2_score == -std::numeric_limits<double>::infinity()) {
                    // -inf means worst possible prediction and is mapped to 0.0
                    regression_loss_.r2_score = 0.0;
                }
            }
#else
            // R^2 score may not be finite
            if (ss_res == 0.0 && ss_tot == 0.0) {
                // NaN means perfect prediction and is mapped to 1.0
                regression_loss_.r2_score = 1.0;
            } else {
                if (ss_tot == 0.0) {
                    // -inf means worst possible prediction and is mapped to 0.0
                    regression_loss_.r2_score = 0.0;
                }
            }
#endif
        }
    }

    // calculate the squared correlation coefficient
    {
        // create helper variables
        double error{ 0.0 };
        double sum_predicted{ 0.0 };
        double sum_correct{ 0.0 };
        double sum_predicted_squared{ 0.0 };
        double sum_correct_squared{ 0.0 };
        double sum_predicted_times_correct{ 0.0 };
        const auto total = static_cast<double>(predicted_label.size());

        // calculate regression score metrics helper variables
#pragma omp parallel for default(none) shared(correct_label, predicted_label) reduction(+ : error, sum_predicted, sum_correct, sum_predicted_squared, sum_correct_squared, sum_predicted_times_correct)
        for (std::size_t i = 0; i < predicted_label.size(); ++i) {
            error += static_cast<double>(predicted_label[i] - correct_label[i]) * static_cast<double>(predicted_label[i] - correct_label[i]);
            sum_predicted += static_cast<double>(predicted_label[i]);
            sum_correct += static_cast<double>(correct_label[i]);
            sum_predicted_squared += static_cast<double>(predicted_label[i] * predicted_label[i]);
            sum_correct_squared += static_cast<double>(correct_label[i] * correct_label[i]);
            sum_predicted_times_correct += static_cast<double>(predicted_label[i] * correct_label[i]);
        }
        regression_loss_.squared_correlation_coefficient = ((total * sum_predicted_times_correct - sum_predicted * sum_correct) * (total * sum_predicted_times_correct - sum_predicted * sum_correct)) / ((total * sum_predicted_squared - sum_predicted * sum_predicted) * (total * sum_correct_squared - sum_correct * sum_correct));
    }
}

/**
 * @brief Output the regression @p report to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the report to
 * @param[in] report the regression_report
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const regression_report &report);
/**
 * @brief Output the @p metric to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the metric to
 * @param[in] metric the metric
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const regression_report::metric &metric);

}  // namespace plssvm

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::regression_report> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::regression_report::metric> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_REGRESSION_REPORT_HPP_

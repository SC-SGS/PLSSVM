/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a classification report returned in the `plssvm::csvm::score` functions.
 */

#ifndef PLSSVM_CLASSIFICATION_REPORT_HPP_
#define PLSSVM_CLASSIFICATION_REPORT_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/igor_utility.hpp"    // plssvm::detail::{has_only_named_args_v, get_value_from_named_parameter}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT, igor::parser, igor::has_unnamed_arguments, igor::has_other_than

#include <algorithm>    // std::find, std::find_if, std::count, std::sort, std::any_of
#include <cstddef>      // std::size_t
#include <iosfwd>       // forward declaration for std::ostream and std::istream
#include <iterator>     // std::distance
#include <set>          // std::set
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair, std::move, std::forward
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Class calculating a classification report (overall accuracy and precision, recall, f1 score, and support per class.
 * @details Calculates the values using an explicit confusion matrix.
 */
class classification_report {
  public:
    /// Create a named argument for the number of floating point digits printed in the classification report.
    static IGOR_MAKE_NAMED_ARGUMENT(digits);
    /// Create a named argument for the zero division behavior: warn (and set to 0.0), 0.0, 1.0, or NaN.
    static IGOR_MAKE_NAMED_ARGUMENT(zero_division);
    /// Create a named argument for the displayed target names in the classification report as `std::vector<std::string>`. Must have the same number of names as there are labels.
    static IGOR_MAKE_NAMED_ARGUMENT(target_names);

    /**
     * @brief Enum class for all possible zero division behaviors when calculating the precision, recall, or F1-score.
     */
    enum class zero_division_behavior {
        /** Print a warning and set the value to 0.0. */
        warn,
        /** Set the value to 0.0. */
        zero,
        /** Set the value to 1.0. */
        one,
        /** Set the value to [`std::numeric_limits<double>::quiet_NaN()`](https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN) */
        nan
    };

    /**
     * @brief Struct encapsulating the different metrics, i.e., precision, recall, f1 score, and support.
     */
    struct metric {
        /// The number of true positives.
        unsigned long long TP{};
        /// The number of true negatives.
        unsigned long long FP{};
        /// The number of false negatives.
        unsigned long long FN{};
        /// The precision calculated as: TP / (TP + FP).
        double precision{};
        /// The recall calculated as: TP / (TP + FN).
        double recall{};
        /// The f1 score calculated as: 2 * (precision * recall) / (precision + recall), i.e., the harmonic mean of precision and recall.
        double f1{};
        /// The number of times the label associated with this metric occurs in the correct label list.
        unsigned long long support{};
    };
    /**
     * @brief Struct encapsulating the different values used for the accuracy metric, i.e., the achieved accuracy (floating point, **not** percent),
     *        the number of correctly predicted labels, and the total number of labels.
     */
    struct accuracy_metric {
        /// The achieved accuracy. A value of 0.95 means an accuracy of 95%.
        double achieved_accuracy{};
        /// The number of correctly predicted labels.
        unsigned long long num_correct{};
        /// The total number of labels.
        unsigned long long num_total{};
    };

    /**
     * @brief Calculates the confusion matrix, classification metrics per class, and global accuracy.
     * @tparam label_type the type of the labels
     * @param[in] correct_label the list of correct labels
     * @param[in] predicted_label the list of predicted labels
     * @param[in] named_args the potential name arguments (digits, zero_division, target_names)
     * @throws plssvm::exception if the @p correct_label or @p predicted_label are empty
     * @throws plssvm::exception if the @p correct_label and @p predicted_label sizes mismatch
     * @throws plssvm::exception if the number of digits to print has been provided but is less or equal to 0
     */
    template <typename label_type, typename... Args>
    classification_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label, Args &&...named_args);

    /**
     * @brief Return the confusion matrix.
     * @return the confusion matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] const aos_matrix<unsigned long long> &confusion_matrix() const noexcept { return confusion_matrix_; }
    /**
     * @brief Return the achieved accuracy.
     * @return the achieved accuracy (`[[nodiscard]]`)
     */
    [[nodiscard]] accuracy_metric accuracy() const noexcept { return accuracy_; }
    /**
     * @brief Get the metrics associated with the provided @p label.
     * @details The metrics are: precision, recall, f1 score, and support.
     * @tparam label_type the type of the label
     * @param[in] label the label to query the metrics for
     * @throws plssvm::exception if the @p label couldn't be found
     * @return the classification report for the specific label (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] metric metric_for_class(const label_type &label) const {
        const auto it = std::find_if(metrics_.cbegin(), metrics_.cend(), [label_str = fmt::format("{}", label)](const auto &p) { return p.first == label_str; });
        if (it == metrics_.cend()) {
            throw exception{ fmt::format("Couldn't find the label \"{}\"!", label) };
        }
        return it->second;
    }

    /**
     * @brief Output the classification @p report to the given output-stream @p out.
     * @details Outputs the metrics in a tabular format.
     * @param[in,out] out the output-stream to write the backend type to
     * @param[in] report the classification_report
     * @return the output-stream
     */
    friend std::ostream &operator<<(std::ostream &out, const classification_report &report);

  private:
    /// The confusion matrix.
    aos_matrix<unsigned long long> confusion_matrix_{};
    /// The metrics for each label: precision, recall, f1 score, and support.
    std::vector<std::pair<std::string, metric>> metrics_;
    /// The global accuracy.
    accuracy_metric accuracy_;

    /// The number of floating point digits printed in the classification report output.
    int output_digits_{ 2 };
    /// Flag, whether the micro average or the accuracy should be printed in the classification report output.
    bool use_micro_average_{ false };
    /// The used zero division behavior.
    zero_division_behavior zero_div_{ zero_division_behavior::warn };
};

namespace detail {

/**
 * @brief Divide the @p dividend by the @p divisor using the @p zero_div zero division behavior if @p divisor is `0`.
 * @param dividend the dividend
 * @param divisor the divisor
 * @param zero_div the zero division behavior
 * @param metric_name the metric name in which the zero division occurred, only used if @p zero_div is equal to `classification_report::zero_division_behavior::warn`
 * @return the quotient (`[[nodiscard]]`)
 */
[[nodiscard]] double sanitize_nan(double dividend, double divisor, classification_report::zero_division_behavior zero_div, std::string_view metric_name);

}  // namespace detail

template <typename label_type, typename... Args>
classification_report::classification_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label, Args &&...named_args) {
    // sanity check for input correct sizes
    if (correct_label.empty()) {
        throw exception{ "The correct labels list must not be empty!" };
    }
    if (predicted_label.empty()) {
        throw exception{ "The predicted labels list must not be empty!" };
    }
    if (correct_label.size() != predicted_label.size()) {
        throw exception{ fmt::format("The number of correct labels ({}) and predicted labels ({}) must be the same!", correct_label.size(), predicted_label.size()) };
    }

    igor::parser parser{ std::forward<Args>(named_args)... };

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(plssvm::classification_report::digits, plssvm::classification_report::zero_division, plssvm::classification_report::target_names),
                  "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(plssvm::classification_report::digits)) {
        // get the value of the provided named parameter
        output_digits_ = detail::get_value_from_named_parameter<decltype(output_digits_)>(parser, plssvm::classification_report::digits);
        // runtime check: the number of digits must be greater than zero!
        if (output_digits_ <= 0) {
            throw exception{ fmt::format("Invalid number of output digits provided! Number of digits must be greater than zero but is {}!", output_digits_) };
        }
    }

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(plssvm::classification_report::zero_division)) {
        // get the value of the provided named parameter
        zero_div_ = detail::get_value_from_named_parameter<decltype(zero_div_)>(parser, plssvm::classification_report::zero_division);
    }

    // initialize confusion matrix
    std::set<label_type> distinct_label(correct_label.cbegin(), correct_label.cend());  // use std::set for a predefined label order
    distinct_label.insert(predicted_label.cbegin(), predicted_label.cend());
    std::vector<label_type> distinct_label_vec(distinct_label.cbegin(), distinct_label.cend());

    // compile time/runtime check: the values must have the correct types
    std::vector<std::string> display_names;
    if constexpr (parser.has(plssvm::classification_report::target_names)) {
        // get the value of the provided named parameter
        display_names = detail::get_value_from_named_parameter<decltype(display_names)>(parser, plssvm::classification_report::target_names);
        // the number of display names must match the number of distinct labels
        if (display_names.size() != distinct_label_vec.size()) {
            throw plssvm::exception{ fmt::format("Provided {} target names, but found {} distinct labels!", display_names.size(), distinct_label_vec.size()) };
        }
    }

    // allocate confusion matrx
    confusion_matrix_ = aos_matrix<unsigned long long>{ distinct_label_vec.size(), distinct_label_vec.size() };

    // function to map a label to its confusion matrix index
    const auto label_to_idx = [&](const label_type &label) -> std::size_t {
        return std::distance(distinct_label_vec.cbegin(), std::find(distinct_label_vec.cbegin(), distinct_label_vec.cend(), label));
    };

    // fill the confusion matrix
    for (typename std::vector<label_type>::size_type i = 0; i < correct_label.size(); ++i) {
        if (detail::contains(distinct_label_vec, correct_label[i]) && detail::contains(distinct_label_vec, predicted_label[i])) {
            ++confusion_matrix_(label_to_idx(correct_label[i]), label_to_idx(predicted_label[i]));
        }
    }

    // calculate the metrics for each label
    for (typename std::vector<label_type>::size_type label_idx = 0; label_idx < distinct_label_vec.size(); ++label_idx) {
        const label_type &label = distinct_label_vec[label_idx];

        // get the display_label -> if target_names is provided use this value, otherwise the actually used label name
        const std::string label_name = display_names.empty() ? fmt::format("{}", label) : display_names[label_idx];

        // calculate TP, FN, and FP
        const unsigned long long TP = confusion_matrix_(label_idx, label_idx);
        unsigned long long FN{ 0 };
        unsigned long long FP{ 0 };
        for (typename std::set<label_type>::size_type i = 0; i < distinct_label_vec.size(); ++i) {
            if (i != label_idx) {
                FN += confusion_matrix_(label_idx, i);
                FP += confusion_matrix_(i, label_idx);
            }
        }
        const double precision = detail::sanitize_nan(static_cast<double>(TP), static_cast<double>(TP + FP), zero_div_, "Precision");
        const double recall = detail::sanitize_nan(static_cast<double>(TP), static_cast<double>(TP + FN), zero_div_, "Recall");
        const double f1 = detail::sanitize_nan(2 * (precision * recall), (precision + recall), zero_div_, "F1-score");
        // add metric results to map
        const metric m{ TP, FP, FN, precision, recall, f1, static_cast<unsigned long long>(std::count(correct_label.cbegin(), correct_label.cend(), label)) };
        metrics_.emplace_back(label_name, m);
    }

    // calculate accuracy
    unsigned long long count{ 0 };
    for (typename std::vector<label_type>::size_type i = 0; i < correct_label.size(); ++i) {
        if (correct_label[i] == predicted_label[i]) {
            ++count;
        }
    }
    accuracy_ = accuracy_metric{ static_cast<double>(count) / static_cast<double>(correct_label.size()), count, correct_label.size() };
}

/**
 * @brief Output the metric @p m to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the metric to
 * @param[in] metric the metric
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const classification_report::metric &metric);
/**
 * @brief Output the accuracy_metric @p accuracy to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the accuracy metric to
 * @param[in] accuracy the accuracy metric
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const classification_report::accuracy_metric &accuracy);
/**
 * @brief Output the zero division behavior @p zero_div to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the zero division behavior to
 * @param[in] zero_div the zero division behavior
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, classification_report::zero_division_behavior zero_div);
/**
 * @brief Use the input-stream @p in to initialize the @p zero_div behavior.
 * @param[in,out] in input-stream to extract the zero division behavior from
 * @param[in] zero_div the zero division behavior
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, classification_report::zero_division_behavior &zero_div);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::classification_report> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<plssvm::classification_report::accuracy_metric> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<plssvm::classification_report::zero_division_behavior> : fmt::ostream_formatter {};

#endif  // PLSSVM_CLASSIFICATION_REPORT_HPP_

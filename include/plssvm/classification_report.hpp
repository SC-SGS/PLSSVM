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

#include <algorithm>  // std::find, std::count
#include <cmath>      // std::isnan
#include <cstddef>    // std::size_t
#include <iosfwd>     // forward declaration for std::ostream and std::istream
#include <iterator>   // std::distance
#include <map>        // std::map
#include <set>        // std::set
#include <string>     // std::string
#include <vector>     // std::vector

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
     * @param[in] named_args the potential name arguments (digits and/or zero_division)
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
     * @return the classification report for the specific label (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] metric metric_for_class(const label_type &label) const { return metrics_.at(fmt::format("{}", label)); }

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
    std::map<std::string, metric> metrics_;
    /// The global accuracy.
    accuracy_metric accuracy_;

    /// The number of floating point digits printed in the classification report output.
    int output_digits_{ 2 };
};

template <typename label_type, typename... Args>
classification_report::classification_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label, Args &&...named_args) {
    PLSSVM_ASSERT(!correct_label.empty(), "The correct labels list may not be empty!");
    PLSSVM_ASSERT(!predicted_label.empty(), "The predicted labels list may not be empty!");

    igor::parser parser{ std::forward<Args>(named_args)... };

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(plssvm::classification_report::digits, plssvm::classification_report::zero_division),
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
    zero_division_behavior zero_div{ zero_division_behavior::warn };
    if constexpr (parser.has(plssvm::classification_report::zero_division)) {
        // get the value of the provided named parameter
        zero_div = detail::get_value_from_named_parameter<decltype(zero_div)>(parser, plssvm::classification_report::zero_division);
    }

    if (correct_label.size() != predicted_label.size()) {
        throw exception{ fmt::format("The number of correct labels ({}) and predicted labels ({}) must be the same!", correct_label.size(), predicted_label.size()) };
    }

    // initialize confusion matrix
    std::set<label_type> distinct_label(correct_label.cbegin(), correct_label.cend());  // use std::Set for a predefined label order
    distinct_label.insert(predicted_label.cbegin(), predicted_label.cend());
    confusion_matrix_ = aos_matrix<unsigned long long>{ distinct_label.size(), distinct_label.size() };

    // function to map a label to its confusion matrix index
    const auto label_to_idx = [&](const label_type &label) -> std::size_t {
        return std::distance(distinct_label.cbegin(), std::find(distinct_label.cbegin(), distinct_label.cend(), label));
    };

    // fill the confusion matrix
    for (typename std::vector<label_type>::size_type i = 0; i < correct_label.size(); ++i) {
        ++confusion_matrix_(label_to_idx(correct_label[i]), label_to_idx(predicted_label[i]));
    }

    // calculate the metrics for each label
    for (const label_type &label : distinct_label) {
        const std::size_t label_idx = label_to_idx(label);
        // calculate TP, FN, and FP
        const unsigned long long TP = confusion_matrix_(label_idx, label_idx);
        unsigned long long FN{ 0 };
        unsigned long long FP{ 0 };
        for (typename std::set<label_type>::size_type i = 0; i < distinct_label.size(); ++i) {
            if (i != label_idx) {
                FN += confusion_matrix_(label_idx, i);
                FP += confusion_matrix_(i, label_idx);
            }
        }
        // calculate precision, recall, and f1 score
        const auto sanitize_nan = [zero_div](const double dividend, const double divisor, const std::string_view metric_name) {
            if (divisor == 0.0) {
                // handle the correct zero division behavior
                switch (zero_div) {
                    case zero_division_behavior::warn:
                        std::clog << metric_name << " is ill-defined and is set to 0.0 in labels with no predicted samples. "
                                                    "Use 'plssvm::classification_report::zero_division' parameter to control this behavior.\n";
                        [[fallthrough]];
                    case zero_division_behavior::zero:
                        return 0.0;
                    case zero_division_behavior::one:
                        return 1.0;
                    case zero_division_behavior::nan:
                        return std::numeric_limits<double>::quiet_NaN();
                }
            } else {
                return dividend / divisor;
            }
        };
        const double precision = sanitize_nan(static_cast<double>(TP), static_cast<double>(TP + FP), "Precision");
        const double recall = sanitize_nan(static_cast<double>(TP), static_cast<double>(TP + FN), "Recall");
        const double f1 = sanitize_nan(2 * (precision * recall), (precision + recall), "F1-score");
        // add metric results to map
        const metric m{ precision, recall, f1, static_cast<unsigned long long>(std::count(correct_label.cbegin(), correct_label.cend(), label)) };
        metrics_.emplace(fmt::format("{}", label), m);
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

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
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <algorithm>  // std::find, std::count
#include <cmath>      // std::isnan
#include <cstddef>    // std::size_t
#include <iosfwd>     // forward declaration for std::ostream
#include <iterator>   // std::distance
#include <map>        // std::map
#include <set>        // std::set
#include <string>     // std::string
#include <vector>     // std::vector

#include "fmt/core.h"  // fmt::format

namespace plssvm {

/**
 * @brief Class calculating a classification report (overall accuracy and precision, recall, f1 score, and support per class.
 * @details Calculates the values using an explicit confusion matrix.
 */
class classification_report {
  public:
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
        std::size_t support{};
    };
    /**
     * @brief Struct encapsulating the different values used for the accuracy metric, i.e., the achieved accuracy (floating point, **not** percent),
     *        the number of correctly predicted labels, and the total number of labels.
     */
    struct accuracy_metric {
        /// The achieved accuracy. A value of 0.95 means an accuracy of 95%.
        double achieved_accuracy{};
        /// The number of correctly predicted labels.
        std::size_t num_correct{};
        /// The total number of labels.
        std::size_t num_total{};
    };

    /**
     * @brief Calculates the confusion matrix, classification metrics per class, and global accuracy.
     * @tparam label_type the type of the labels
     * @param[in] correct_label the list of correct labels
     * @param[in] predicted_label the list of predicted labels
     * @throws plssvm::exception if the @p correct_label and @p predicted_label sizes mismatch
     */
    template <typename label_type>
    classification_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label);

    /**
     * @brief Return the confusion matrix.
     * @return the confusion matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::vector<std::size_t>> &confusion_matrix() const noexcept { return confusion_matrix_; }
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
    std::vector<std::vector<std::size_t>> confusion_matrix_{};
    /// The metrics for each label: precision, recall, f1 score, and support.
    std::map<std::string, metric> metrics_;
    /// The global accuracy.
    accuracy_metric accuracy_;
};

template <typename label_type>
classification_report::classification_report(const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label) {
    PLSSVM_ASSERT(!correct_label.empty(), "The correct labels list may not be empty!");
    PLSSVM_ASSERT(!predicted_label.empty(), "The predicted labels list may not be empty!");

    if (correct_label.size() != predicted_label.size()) {
        throw exception{ fmt::format("The number of correct labels ({}) and predicted labels ({}) must be the same!", correct_label.size(), predicted_label.size()) };
    }

    // initialize confusion matrix
    std::set<label_type> distinct_label(correct_label.cbegin(), correct_label.cend());
    distinct_label.insert(predicted_label.cbegin(), predicted_label.cend());
    confusion_matrix_ = std::vector<std::vector<std::size_t>>(distinct_label.size(), std::vector<std::size_t>(distinct_label.size()));

    // function to map a label to its confusion matrix index
    const auto label_to_idx = [&](const label_type &label) -> std::size_t {
        return std::distance(distinct_label.cbegin(), std::find(distinct_label.cbegin(), distinct_label.cend(), label));
    };

    // fill the confusion matrix
    for (std::size_t i = 0; i < correct_label.size(); ++i) {
        ++confusion_matrix_[label_to_idx(correct_label[i])][label_to_idx(predicted_label[i])];
    }

    // calculate the metrics for each label
    for (const label_type &label : distinct_label) {
        const std::size_t label_idx = label_to_idx(label);
        // calculate TP, FN, FP, and TN
        const std::size_t TP = confusion_matrix_[label_idx][label_idx];
        std::size_t FN = 0;
        std::size_t FP = 0;
        for (std::size_t i = 0; i < distinct_label.size(); ++i) {
            if (i != label_idx) {
                FN += confusion_matrix_[label_idx][i];
                FP += confusion_matrix_[i][label_idx];
            }
        }
        // const std::size_t TN = correct_label.size() - TP - FN - FP;
        // calculate precision, recall, and f1 score
        const auto sanitize_nan = [](const double val) {
            return std::isnan(val) ? 0.0 : val;
        };
        const double precision = sanitize_nan(static_cast<double>(TP) / static_cast<double>(TP + FP));
        const double recall = sanitize_nan(static_cast<double>(TP) / static_cast<double>(TP + FN));
        const double f1 = sanitize_nan(2 * (precision * recall) / (precision + recall));
        // add metric results to map
        const metric m{ precision, recall, f1, static_cast<std::size_t>(std::count(correct_label.cbegin(), correct_label.cend(), label)) };
        metrics_.emplace(fmt::format("{}", label), m);
    }

    // calculate accuracy
    std::size_t count{ 0 };
    for (std::size_t i = 0; i < correct_label.size(); ++i) {
        if (correct_label[i] == predicted_label[i]) {
            ++count;
        }
    }
    accuracy_ = accuracy_metric{ static_cast<double>(count) / static_cast<double>(correct_label.size()), count, correct_label.size() };
}

std::ostream &operator<<(std::ostream &out, const classification_report::accuracy_metric &accuracy);

}  // namespace plssvm

template <> struct fmt::formatter<plssvm::classification_report> : fmt::ostream_formatter {};
template <> struct fmt::formatter<plssvm::classification_report::accuracy_metric> : fmt::ostream_formatter {};

#endif  // PLSSVM_CLASSIFICATION_REPORT_HPP_

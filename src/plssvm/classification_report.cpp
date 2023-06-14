/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/classification_report.hpp"

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::max, std::sort
#include <cstddef>    // std::size_t
#include <ostream>    // std::ostream, std::endl
#include <string>     // std::string
#include <utility>    // std::pair

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const classification_report::accuracy_metric &accuracy) {
    return out << fmt::format("Accuracy = {:.2f}% ({}/{})", accuracy.achieved_accuracy * 100, accuracy.num_correct, accuracy.num_total);
}

std::ostream &operator<<(std::ostream &out, const classification_report &report) {
    // calculate the maximum size of the label for better table alignment
    std::size_t max_label_string_size = 12;  // weighted avg = 12 characters
    for (const auto &[key, val] : report.metrics_) {
        max_label_string_size = std::max(max_label_string_size, key.size());
    }

    // confusion matrix not printed due to potential size

    // print metrics
    out << fmt::format("{:>{}}   precision   recall     f1   support\n", "", max_label_string_size);
    // used for sorting the table output
    std::vector<std::pair<double, std::string>> table_rows;
    table_rows.reserve(report.metrics_.size());
    classification_report::metric macro_avg{};
    classification_report::metric weighted_avg{};
    for (const auto &[key, val] : report.metrics_) {
        table_rows.emplace_back(val.precision, fmt::format("{:>{}}        {:.2f}     {:.2f}   {:.2f}   {:>7}\n", key, max_label_string_size, val.precision, val.recall, val.f1, val.support));
        macro_avg.precision += val.precision;
        weighted_avg.precision += val.precision * static_cast<double>(val.support);
        macro_avg.recall += val.recall;
        weighted_avg.recall += val.recall * static_cast<double>(val.support);
        macro_avg.f1 += val.f1;
        weighted_avg.f1 += val.f1 * static_cast<double>(val.support);
        macro_avg.support += val.support;
        weighted_avg.support += val.support;
    }
    // calculate macro average and weighted average
    macro_avg.precision /= static_cast<double>(report.metrics_.size());
    weighted_avg.precision /= static_cast<double>(weighted_avg.support);
    macro_avg.recall /= static_cast<double>(report.metrics_.size());
    weighted_avg.recall /= static_cast<double>(weighted_avg.support);
    macro_avg.f1 /= static_cast<double>(report.metrics_.size());
    weighted_avg.f1 /= static_cast<double>(weighted_avg.support);

    // sort the table rows in increasing order of the precision
    std::sort(table_rows.begin(), table_rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
    });
    // output sorted metrics
    for (const auto &[key, row] : table_rows) {
        out << row;
    }
    out << '\n';
    // print accuracy and average metrics
    out << fmt::format("{:>{}}                        {:.2f}   {:>7}\n", "accuracy", max_label_string_size, report.accuracy_.achieved_accuracy, report.accuracy_.num_total);
    out << fmt::format("{:>{}}        {:.2f}     {:.2f}   {:.2f}   {:>7}\n", "macro avg", max_label_string_size, macro_avg.precision, macro_avg.recall, macro_avg.f1, macro_avg.support);
    out << fmt::format("{:>{}}        {:.2f}     {:.2f}   {:.2f}   {:>7}\n\n", "weighted avg", max_label_string_size, weighted_avg.precision, weighted_avg.recall, weighted_avg.f1, weighted_avg.support);
    out << report.accuracy_ << std::endl;
    return out;
}

}  // namespace plssvm
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
    std::size_t max_label_string_size = 5;  // class = 5 characters
    for (const auto &[key, val] : report.metrics_) {
        max_label_string_size = std::max(max_label_string_size, key.size());
    }

    // confusion matrix not printed due to potential size

    // TODO: make output look like sklearn.metrics.classification_report
    // print metrics
    out << fmt::format("{:>{}}   precision   recall   f1     support\n", "class", max_label_string_size);
    // used for sorting the table output
    std::vector<std::pair<double, std::string>> table_rows;
    table_rows.reserve(report.metrics_.size());
    for (const auto &[key, val] : report.metrics_) {
        table_rows.emplace_back(val.precision, fmt::format("{:>{}}   {:.2f}        {:.2f}     {:.2f}   {}\n", key, max_label_string_size, val.precision, val.recall, val.f1, val.support));
    }
    // sort the table rows in increasing order of the precision
    std::sort(table_rows.begin(), table_rows.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
    });
    // output sorted metrics
    for (const auto &[key, row] : table_rows) {
        out << row;
    }
    out << std::endl;

    // print accuracy
    out << report.accuracy_;
    return out;
}

}  // namespace plssvm
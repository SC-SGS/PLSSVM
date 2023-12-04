/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/classification_report.hpp"

#include "plssvm/detail/logging.hpp"         // plssvm::detail::log
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include "fmt/format.h"  // fmt::format

#include <algorithm>    // std::max, std::sort
#include <cstddef>      // std::size_t
#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <limits>       // std::numeric_limits::quiet_NaN
#include <ostream>      // std::ostream, std::clog, std::endl
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair
#include <vector>       // std::vector

namespace plssvm {

namespace detail {

double sanitize_nan(const double dividend, const double divisor, const classification_report::zero_division_behavior zero_div, const std::string_view metric_name) {
    if (divisor == 0.0) {
        // handle the correct zero division behavior
        switch (zero_div) {
            case classification_report::zero_division_behavior::warn:
                detail::log(verbosity_level::full,
                            "{} is ill-defined and is set to 0.0 in labels with no predicted samples. "
                            "Use 'plssvm::classification_report::zero_division' parameter to control this behavior.\n",
                            metric_name);
                [[fallthrough]];
            case classification_report::zero_division_behavior::zero:
                return 0.0;
            case classification_report::zero_division_behavior::one:
                return 1.0;
            case classification_report::zero_division_behavior::nan:
                return std::numeric_limits<double>::quiet_NaN();
        }
    }
    return dividend / divisor;
}

}  // namespace detail

std::ostream &operator<<(std::ostream &out, const classification_report &report) {
    // calculate the maximum size of the label for better table alignment
    std::size_t max_label_string_size = 12;  // weighted avg = 12 characters
    for (const auto &[key, val] : report.metrics_) {
        max_label_string_size = std::max(max_label_string_size, key.size());
    }

    // confusion matrix not printed due to potential size

    // print metrics
    out << fmt::format("{0:>{1}}{0:>{2}}precision  {0:>{2}}recall{0:>{2}}f1-score   support\n\n", "", max_label_string_size, report.output_digits_);
    // used for sorting the table output
    std::vector<std::string> table_rows;
    table_rows.reserve(report.metrics_.size());
    unsigned long long micro_avg_TP_sum{};
    unsigned long long micro_avg_FP_sum{};
    unsigned long long micro_avg_FN_sum{};
    classification_report::metric micro_avg{};
    classification_report::metric macro_avg{};
    classification_report::metric weighted_avg{};
    for (const auto &[key, val] : report.metrics_) {
        out << fmt::format("{1:>{2}}       {3:.{0}f}      {4:.{0}f}      {5:.{0}f}   {6:>7}\n", report.output_digits_, key, max_label_string_size, val.precision, val.recall, val.f1, val.support);

        // micro average
        micro_avg_TP_sum += val.TP;
        micro_avg_FP_sum += val.FP;
        micro_avg_FN_sum += val.FN;
        micro_avg.support += val.support;

        // macro average
        macro_avg.precision += val.precision;
        macro_avg.recall += val.recall;
        macro_avg.f1 += val.f1;
        macro_avg.support += val.support;

        // weighted average
        weighted_avg.precision += val.precision * static_cast<double>(val.support);
        weighted_avg.recall += val.recall * static_cast<double>(val.support);
        weighted_avg.f1 += val.f1 * static_cast<double>(val.support);
        weighted_avg.support += val.support;
    }
    // calculate micro, macro, and weighted averages
    micro_avg.precision = detail::sanitize_nan(static_cast<double>(micro_avg_TP_sum), static_cast<double>(micro_avg_TP_sum + micro_avg_FP_sum), report.zero_div_, "Precision");
    micro_avg.recall = detail::sanitize_nan(static_cast<double>(micro_avg_TP_sum), static_cast<double>(micro_avg_TP_sum + micro_avg_FN_sum), report.zero_div_, "Recall");
    micro_avg.f1 = detail::sanitize_nan(2 * micro_avg.precision * micro_avg.recall, micro_avg.precision + micro_avg.recall, report.zero_div_, "F1-Score");
    macro_avg.precision /= static_cast<double>(report.metrics_.size());
    macro_avg.recall /= static_cast<double>(report.metrics_.size());
    macro_avg.f1 /= static_cast<double>(report.metrics_.size());
    weighted_avg.precision /= static_cast<double>(weighted_avg.support);
    weighted_avg.recall /= static_cast<double>(weighted_avg.support);
    weighted_avg.f1 /= static_cast<double>(weighted_avg.support);

    out << '\n';
    // print accuracy and average metrics
    if (!report.use_micro_average_) {
        out << fmt::format("{1:>{2}}                       {3:>{4}}{5:.{0}f}   {6:>7}\n", report.output_digits_, "accuracy", max_label_string_size, "", 2 * report.output_digits_, report.accuracy_.achieved_accuracy, report.accuracy_.num_total);
    } else {
        out << fmt::format("{1:>{2}}       {3:.{0}f}      {4:.{0}f}      {5:.{0}f}   {6:>7}\n", report.output_digits_, "micro avg", max_label_string_size, micro_avg.precision, micro_avg.recall, micro_avg.f1, micro_avg.support);
    }
    out << fmt::format("{1:>{2}}       {3:.{0}f}      {4:.{0}f}      {5:.{0}f}   {6:>7}\n", report.output_digits_, "macro avg", max_label_string_size, macro_avg.precision, macro_avg.recall, macro_avg.f1, macro_avg.support);
    out << fmt::format("{1:>{2}}       {3:.{0}f}      {4:.{0}f}      {5:.{0}f}   {6:>7}\n\n", report.output_digits_, "weighted avg", max_label_string_size, weighted_avg.precision, weighted_avg.recall, weighted_avg.f1, weighted_avg.support);
    out << report.accuracy_ << std::endl;
    return out;
}

std::ostream &operator<<(std::ostream &out, const classification_report::metric &metric) {
    return out << fmt::format(
               "TP: {}\n"
               "FP: {}\n"
               "FN: {}\n"
               "precision: {}\n"
               "recall:    {}\n"
               "f1-score:  {}\n"
               "support:   {}",
               metric.TP,
               metric.FP,
               metric.FN,
               metric.precision,
               metric.recall,
               metric.f1,
               metric.support);
}

std::ostream &operator<<(std::ostream &out, const classification_report::accuracy_metric &accuracy) {
    return out << fmt::format("Accuracy = {:.2f}% ({}/{})", accuracy.achieved_accuracy * 100, accuracy.num_correct, accuracy.num_total);
}

std::ostream &operator<<(std::ostream &out, const classification_report::zero_division_behavior zero_div) {
    switch (zero_div) {
        case classification_report::zero_division_behavior::warn:
            return out << "warn";
        case classification_report::zero_division_behavior::zero:
            return out << "0.0";
        case classification_report::zero_division_behavior::one:
            return out << "1.0";
        case classification_report::zero_division_behavior::nan:
            return out << "nan";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, classification_report::zero_division_behavior &zero_div) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "warn") {
        zero_div = classification_report::zero_division_behavior::warn;
    } else if (str == "zero" || str == "0.0") {
        zero_div = classification_report::zero_division_behavior::zero;
    } else if (str == "one" || str == "1.0") {
        zero_div = classification_report::zero_division_behavior::one;
    } else if (str == "nan") {
        zero_div = classification_report::zero_division_behavior::nan;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm
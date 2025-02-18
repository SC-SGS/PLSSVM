/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/regression_report.hpp"

#include "fmt/format.h"  // fmt::format

#include <ostream>  // std::ostream, std::endl

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const regression_report &report) {
    return out << report.loss();
}

std::ostream &operator<<(std::ostream &out, const regression_report::metric &metric) {
    return out << fmt::format(
               "Explained variance score:        {}\n"
               "Mean absolute error:             {}\n"
               "Mean squared error:              {}\n"
               "R^2 score:                       {}\n"
               "Squared correlation coefficient: {}",
               metric.explained_variance_score,
               metric.mean_absolute_error,
               metric.mean_squared_error,
               metric.r2_score,
               metric.squared_correlation_coefficient);
}

}  // namespace plssvm

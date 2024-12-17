/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a data set class encapsulating all data points, features, and potential labels for the regression task.
 */

#ifndef PLSSVM_DATA_SET_REGRESSION_DATA_SET_HPP_
#define PLSSVM_DATA_SET_REGRESSION_DATA_SET_HPP_
#pragma once

#include "plssvm/data_set/data_set.hpp"

namespace plssvm {

class csvr;

template <typename T = int>
class regression_data_set : public data_set<T> {
    using base_data_set = data_set<T>;

    // TODO: further constraint label type -> only integer and floating points?

  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using base_data_set::label_type;
    /// An unsigned integer type.
    using base_data_set::size_type;

    using svm_fit_type = ::plssvm::csvr;

    // forward constructors simply to base class
    template <typename... Args>
    explicit regression_data_set(Args &&...args) :
        base_data_set{ std::forward<Args>(args)... } { }
};

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_REGRESSION_DATA_SET_HPP_

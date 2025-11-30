/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a min max data scaler usable in the data set classes.
 */

#ifndef PLSSVM_DATA_SET_MIN_MAX_SCALER_HPP_
#define PLSSVM_DATA_SET_MIN_MAX_SCALER_HPP_
#pragma once

#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::min_max_scaler_exception
#include "plssvm/matrix.hpp"                               // plssvm::matrix, plssvm::layout_type
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::min, std::max, std::sort, std::adjacent_find
#include <chrono>     // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <limits>     // std::numeric_limits::{max, lowest}
#include <optional>   // std::optional, std::make_optional, std::nullopt
#include <string>     // std::string
#include <utility>    // std::pair
#include <vector>     // std::vector

namespace plssvm {

/**
 * @brief Implements all necessary data and functions needed for scaling a plssvm::data_set to an user-defined range [lower, upper].
 */
class min_max_scaler {
  public:
    /**
     * @brief The calculated or read feature-wise scaling factors.
     * @details Note that the feature indices are zero-based and not one-based.
     */
    struct factors {
        /// The used size type.
        using size_type = std::size_t;

        /**
         * @brief Default construct new scaling factors.
         */
        factors() = default;

        /**
         * @brief Construct new scaling factors struct with the provided values.
         * @param[in] feature_index the feature index for which the bounds are valid
         * @param[in] lower_bound the lowest value of the feature @p feature_index for all data points
         * @param[in] upper_bound the maximum value of the feature @p feature_index for all data points
         */
        factors(const size_type feature_index, const real_type lower_bound, const real_type upper_bound) :
            feature{ feature_index },
            lower{ lower_bound },
            upper{ upper_bound } { }

        /// The feature index for which the scaling factors are valid.
        size_type feature{};
        /// The lowest value of the @p feature for all data points.
        real_type lower{};
        /// The maximum value of the @p feature for all data points.
        real_type upper{};
    };

    /**
     * @brief Create a new scaling class that can be used to scale all features of a data set to the interval [lower, upper].
     * @param[in] lower the lower bound value of all features
     * @param[in] upper the upper bound value of all features
     * @throws plssvm::data_set_exception if lower is greater or equal than upper
     */
    min_max_scaler(real_type lower, real_type upper);
    /**
     * @brief Create a new scaling class that can be used to scale all features of a data set to the interval [lower, upper].
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] lower the lower bound value of all features
     * @param[in] upper the upper bound value of all features
     * @throws plssvm::data_set_exception if lower is greater or equal than upper
     */
    min_max_scaler(mpi::communicator comm, real_type lower, real_type upper);

    /**
     * @brief Read the scaling interval and factors from the provided file @p filename.
     * @param[in] filename the filename to read the scaling information from
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by the plssvm::detail::io::parse_scaling_factors function
     */
    min_max_scaler(const std::string &filename);  // NOLINT: can't be explicit due to the data_set_variant
    /**
     * @brief Read the scaling interval and factors from the provided file @p filename.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the filename to read the scaling information from
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by the plssvm::detail::io::parse_scaling_factors function
     */
    min_max_scaler(mpi::communicator comm, const std::string &filename);

    /**
     * @brief Save the scaling factors to the file @p filename.
     * @param[in] filename the file to save the scaling factors to
     * @throws plssvm::data_set_exception if no scaling factors are available
     */
    void save(const std::string &filename) const;

    /**
     * @brief Scale the feature values of the @p data to the provided range.
     * @details Scales all data points feature wise, i.e., one scaling factor is responsible, e.g., for the first feature of **all** data points. <br>
     *          Scaling a data value \f$x\f$ to the range \f$[a, b]\f$ is done with the formular:
     *          \f$x_{scaled} = a + (b - a) \cdot \frac{x - min(x)}{max(x) - min(x)}\f$
     * @param[in,out] data the data to scale
     * @throws plssvm::data_set_exception if more scaling factors than features are present
     * @throws plssvm::data_set_exception if the largest scaling factor index is larger than the number of features
     * @throws plssvm::data_set_exception if for any feature more than one scaling factor is present
     */
    template <layout_type layout>
    void scale(plssvm::matrix<real_type, layout> &data);

    /**
     * @brief Get the scaling interval. After scaling, all feature values are scaled to [lower, upper].
     * @return { lowest value, largest value } (`[[nodiscard]]`)
     */
    [[nodiscard]] std::pair<real_type, real_type> scaling_interval() const noexcept {
        return scaling_interval_;
    }

    /**
     * @brief Get the scaling factors. If nothing has been scaled yet, returns a std::nullopt.
     * @return an std::optional containing the scaling factors per feature (`[[nodiscard]]`)
     */
    [[nodiscard]] std::optional<std::vector<factors>> scaling_factors() const {
        if (scaling_factors_.empty()) {
            return std::nullopt;  // nothing scaled yet
        } else {
            return std::make_optional(scaling_factors_);
        }
    }

    /**
     * @brief Get the associated MPI communicator.
     * @return the MPI communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] const mpi::communicator &communicator() const noexcept {
        return comm_;
    }

  private:
    /// The user-provided scaling interval. After scaling, all feature values are scaled to [lower, upper].
    std::pair<real_type, real_type> scaling_interval_;
    /// The scaling factors for all features.
    std::vector<factors> scaling_factors_;

    /// The used MPI communicator.
    mpi::communicator comm_;
};

template <layout_type layout>
void min_max_scaler::scale(plssvm::matrix<real_type, layout> &data) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    using size_type = typename plssvm::matrix<real_type, layout>::size_type;
    const size_type num_data_points = data.num_rows();
    const size_type num_features = data.num_cols();

    // unpack scaling interval pair
    const real_type lower = scaling_interval_.first;
    const real_type upper = scaling_interval_.second;

    // calculate scaling factors if necessary, use provided ones otherwise
    if (scaling_factors_.empty()) {
        // calculate feature-wise min/max values for scaling
        for (size_type feature = 0; feature < num_features; ++feature) {
            real_type min_value = std::numeric_limits<real_type>::max();
            real_type max_value = std::numeric_limits<real_type>::lowest();

// calculate min/max values of all data points at the specific feature
#pragma omp parallel for default(shared) firstprivate(feature) reduction(min : min_value) reduction(max : max_value)
            for (size_type data_point = 0; data_point < num_data_points; ++data_point) {
                min_value = std::min(min_value, data(data_point, feature));
                max_value = std::max(max_value, data(data_point, feature));
            }

            // add scaling factor only if min_value != 0.0 AND max_value != 0.0
            if (!(min_value == real_type{ 0.0 } && max_value == real_type{ 0.0 })) {
                scaling_factors_.emplace_back(feature, min_value, max_value);
            }
        }
    } else {
        // the number of scaling factors may not exceed the number of features
        if (scaling_factors_.size() > num_features) {
            throw min_max_scaler_exception{ fmt::format("Need at most as much scaling factors as features in the data set are present ({}), but {} were given!", num_features, scaling_factors_.size()) };
        }
        // sort vector
        const auto scaling_factors_comp_less = [](const factors &lhs, const factors &rhs) { return lhs.feature < rhs.feature; };
        std::sort(scaling_factors_.begin(), scaling_factors_.end(), scaling_factors_comp_less);
        // check whether the biggest feature index is smaller than the number of features
        if (scaling_factors_.back().feature >= num_features) {
            throw min_max_scaler_exception{ fmt::format("The maximum scaling feature index most not be greater or equal than {}, but is {}!", num_features, scaling_factors_.back().feature) };
        }
        // check that there are no duplicate entries
        const auto scaling_factors_comp_eq = [](const factors &lhs, const factors &rhs) { return lhs.feature == rhs.feature; };
        const auto iter = std::adjacent_find(scaling_factors_.begin(), scaling_factors_.end(), scaling_factors_comp_eq);
        if (iter != scaling_factors_.end()) {
            throw min_max_scaler_exception{ fmt::format("Found more than one scaling factor for the feature index {}!", iter->feature) };
        }
    }

// scale values
#pragma omp parallel for default(shared) firstprivate(lower, upper)
    for (size_type i = 0; i < scaling_factors_.size(); ++i) {
        // extract feature-wise min and max values
        const factors factor = scaling_factors_[i];
        // scale data values
        for (size_type data_point = 0; data_point < num_data_points; ++data_point) {
            data(data_point, factor.feature) = lower + (upper - lower) * (data(data_point, factor.feature) - factor.lower) / (factor.upper - factor.lower);
        }
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                comm_,
                "Scaled the data set to the range [{}, {}] in {}.\n",
                detail::tracking::tracking_entry{ "data_set_scale", "lower", lower },
                detail::tracking::tracking_entry{ "data_set_scale", "upper", upper },
                detail::tracking::tracking_entry{ "data_set_scale", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });
}

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_MIN_MAX_SCALER_HPP_

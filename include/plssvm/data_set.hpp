/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a data set class encapsulating all data points and potential features.
 */

#ifndef PLSSVM_DATA_SET_HPP_
#define PLSSVM_DATA_SET_HPP_
#pragma once

#include "plssvm/constants.hpp"                          // plssvm::verbose
#include "plssvm/detail/cmd/parameter_predict.hpp"       // plssvm::detail::cmd::parameter_predict
#include "plssvm/detail/cmd/parameter_scale.hpp"         // plssvm::detail::cmd::parameter_scale
#include "plssvm/detail/cmd/parameter_train.hpp"         // plssvm::detail::cmd::parameter_train
#include "plssvm/detail/io/arff_parsing.hpp"             // plssvm::detail::io::{read_libsvm_data, write_libsvm_data}
#include "plssvm/detail/io/file_reader.hpp"              // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_parsing.hpp"           // plssvm::detail::io::{read_arff_header, read_arff_data, write_arff_header, write_arff_data}
#include "plssvm/detail/io/scaling_factors_parsing.hpp"  // plssvm::detail::io::{read_scaling_factors}
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::ends_with
#include "plssvm/exceptions/exceptions.hpp"              // plssvm::exception
#include "plssvm/file_format_types.hpp"                  // plssvm::file_format_type

#include "fmt/chrono.h"   // directly output std::chrono times via fmt
#include "fmt/core.h"     // fmt::format, fmt::print
#include "fmt/os.h"       // fmt::ostream
#include "fmt/ostream.h"  // directly output objects with operator<< overload via fmt

#include <algorithm>    // std::all_of, std::max, std::min
#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast, millisecond}
#include <cstddef>      // std::size_t
#include <functional>   // std::reference_wrapper, std::cref
#include <iostream>     // std::cout, std::endl
#include <limits>       // std::numeric_limits
#include <map>          // std::map
#include <memory>       // std::shared_ptr, std::make_shared
#include <optional>     // std::optional, std::make_optional, std::nullopt
#include <set>          // std::set
#include <string>       // std::string
#include <tuple>        // std::tie
#include <type_traits>  // std::is_same_v, std::is_arithmetic_v
#include <utility>      // std::move, std::pair
#include <variant>      // std::variant
#include <vector>       // std::vector

namespace plssvm {

template <typename T>
using optional_ref = std::optional<std::reference_wrapper<T>>;

template <typename T, typename U = int>
class data_set {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The first template type can only be 'float' or 'double'!");
    static_assert(std::is_arithmetic_v<U> || std::is_same_v<U, std::string>, "The second template type can only be an arithmetic type or 'std::string'!");
    // because std::vector<bool> is evil
    static_assert(!std::is_same_v<U, bool>, "The second template type must NOT be 'bool'!");

    // plssvm::model needs the default constructor
    template <typename, typename>
    friend class model;

  public:
    using real_type = T;
    using label_type = U;
    using size_type = std::size_t;

    class scaling;

    class label_mapper;

    explicit data_set(const std::string &filename);
    data_set(const std::string &filename, file_format_type format);
    data_set(const std::string &filename, scaling scale_parameter);
    data_set(const std::string &filename, file_format_type format, scaling scale_parameter);

    explicit data_set(std::vector<std::vector<real_type>> &&X);
    data_set(std::vector<std::vector<real_type>> &&X, std::vector<label_type> &&y);
    data_set(std::vector<std::vector<real_type>> &&X, scaling scale_parameter);
    data_set(std::vector<std::vector<real_type>> &&X, std::vector<label_type> &&y, scaling scale_parameter);

    // save the data set in the given format
    void save(const std::string &filename, file_format_type format) const;

    [[nodiscard]] const std::vector<std::vector<real_type>> &data() const noexcept { return *X_ptr_; }
    [[nodiscard]] bool has_labels() const noexcept { return labels_ptr_ != nullptr; }
    [[nodiscard]] optional_ref<const std::vector<real_type>> mapped_labels() const noexcept;
    [[nodiscard]] optional_ref<const std::vector<label_type>> labels() const noexcept;

    [[nodiscard]] label_type label_from_mapped_value(real_type val) const;

    [[nodiscard]] size_type num_labels() const noexcept { return mapping_ != nullptr ? mapping_->num_mappings() : 0; }
    [[nodiscard]] size_type num_data_points() const noexcept { return num_data_points_; }
    [[nodiscard]] size_type num_features() const noexcept { return num_features_; }

    [[nodiscard]] optional_ref<const label_mapper> mapping() const noexcept;

    [[nodiscard]] bool is_scaled() const noexcept { return scale_parameters_ != nullptr; }
    [[nodiscard]] optional_ref<const scaling> scaling_factors() const noexcept;

  private:
    data_set();

    void create_mapping();
    void scale();

    void write_libsvm_file(const std::string &filename) const;
    void write_arff_file(const std::string &filename) const;
    void read_file(const std::string &filename, file_format_type format);
    void read_libsvm_file(const std::string &filename);
    void read_arff_file(const std::string &filename);

    std::shared_ptr<std::vector<std::vector<real_type>>> X_ptr_{ nullptr };
    std::shared_ptr<std::vector<real_type>> y_ptr_{ nullptr };
    std::shared_ptr<std::vector<label_type>> labels_ptr_{ nullptr };

    size_type num_data_points_{ 0 };
    size_type num_features_{ 0 };

    std::shared_ptr<const label_mapper> mapping_{ nullptr };

    std::shared_ptr<scaling> scale_parameters_{ nullptr };
};

////////////////////////////////////////////////////////////////////////////////
////                          scaling nested-class                          ////
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Implements all necessary data needed for scaling a data set to an user-defined range.
 * @tparam T the data type of the data set
 * @tparam U the label type of the data set
 */
template <typename T, typename U>
class data_set<T, U>::scaling {
  public:
    /**
     * @brief The calculated or read feature-wise scaling factors.
     */
    struct factors {
        /**
         * @brief Default construct new scaling factors.
         */
        factors() = default;
        /**
         * @brief Construct new scaling factors with the provided values.
         * @param[in] p_feature the feature index for which the bounds are valid
         * @param[in] p_lower the lowest value of the feature @p p_feature for all data points
         * @param[in] p_upper the maximum value of the feature @p p_feature for all data points
         */
        factors(const size_type p_feature, const real_type p_lower, const real_type p_upper) :
            feature{ p_feature }, lower{ p_lower }, upper{ p_upper } {}

        /// The feature index for which the scaling factors are valid.
        size_type feature{};
        /// The lowest value of the feature for all data points.
        real_type lower{};
        /// The maximum value of the feature for all data points.
        real_type upper{};
    };

    /**
     * @brief Create a new scaling class that can be used to scale all features of a data set to the interval [lower, upper].
     * @param[in] lower the lower bound of all features
     * @param[in] upper the upper bound of all features
     */
    scaling(real_type lower, real_type upper);
    /**
     * @brief Read the scaling interval and factors from the provided file @p filename.
     * @param[in] filename the filename to read the scaling information from
     */
    scaling(const std::string &filename);

    /**
     * @brief Save the scaling factors to the file @p filename.
     * @param[in] filename the file to save the scaling factors to
     * @throws plssvm::data_set_exception if no scaling factors are available
     */
    void save(const std::string &filename) const;

    /// The user-provided scaling interval. After scaling, all feature values are scaled to [lower, upper].
    std::pair<real_type, real_type> scaling_interval{};
    /// The scaling factors for all features.
    std::vector<factors> scaling_factors{};
};

template <typename T, typename U>
data_set<T, U>::scaling::scaling(const real_type lower, const real_type upper) :
    scaling_interval{ std::make_pair(lower, upper) } {}

template <typename T, typename U>
data_set<T, U>::scaling::scaling(const std::string &filename) {
    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // read scaling values from file
    std::tie(scaling_interval, scaling_factors) = detail::io::parse_scaling_factors<real_type, factors>(reader);
}

template <typename T, typename U>
void data_set<T, U>::scaling::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // write scaling values to file
    detail::io::write_scaling_factors(filename, scaling_interval, scaling_factors);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << fmt::format("Write {} scaling factors in {} to the file '{}'.",
                                 scaling_factors.size(),
                                 std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                                 filename)
                  << std::endl;
    }
}

////////////////////////////////////////////////////////////////////////////////
////                       label mapper nested-class                        ////
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Implements all necessary functionality to map arbitrary labels to labels usable by the C-SVMs.
 * @tparam T the data type of the data set
 * @tparam U the label type of the data set
 */
template <typename T, typename U>
class data_set<T, U>::label_mapper {
  public:
    /**
     * @brief Create a mapping from all labels to { -1 , 1 } and vice versa.
     * @param[in] labels the labels to map
     * @note Currently only binary classification is supported, i.e., only two different labels may be provided!
     * @throws plssvm::data_set_exception only two different labels are currently supported
     */
    explicit label_mapper(const std::vector<label_type> &labels);

    /**
     * @brief Given the original label value, return the mapped label value.
     * @param[in] label the original label value
     * @throws plssvm::data_set_exception if the original label value does not exist in this mapping
     * @return the mapped label value (`[[nodiscard]]`)
     */
    [[nodiscard]] const real_type &get_mapped_value_by_label(const label_type &label) const;
    /**
     * @brief Given the mapped label value, return the original label value.
     * @param[in] mapped_value the mapped label value
     * @throws plssvm::data_set_exception if the mapped label value does not exist in this mapping
     * @return the original label value (`[[nodiscard]]`)
     */
    [[nodiscard]] const label_type &get_label_by_mapped_value(const real_type &mapped_value) const;
    /**
     * @brief Returns the number of valid mappings. This is equivalent to the number of different labels.
     * @return the number of valid mapping entries (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_mappings() const noexcept;
    /**
     * @brief Return the original labels of the current data set.
     * @return the original labels (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<label_type> labels() const;

  private:
    /// A mapping from the label to its mapped value, i.e., { -1 , 1 }.
    std::map<label_type, real_type> label_to_mapped_{};
    /// A mapping from the mapped value, i.e., { -1 , 1 } to the original label value.
    std::map<real_type, label_type> mapped_to_label_{};
};

template <typename T, typename U>
data_set<T, U>::data_set::label_mapper::label_mapper(const std::vector<label_type> &labels) {
    // we are only interested in unique labels
    std::set<label_type> unique_labels(labels.begin(), labels.end());
    // currently, only two different labels are supported
    if (unique_labels.size() != 2) {
        throw data_set_exception{ fmt::format("Currently only binary classification is supported, but {} different labels were given!", unique_labels.size()) };
    }
    // create mapping
    // first label
    auto iter = unique_labels.begin();
    label_to_mapped_[*iter] = -1;
    mapped_to_label_[-1] = *iter;
    // second label
    ++iter;
    label_to_mapped_[*iter] = +1;
    mapped_to_label_[+1] = *iter;
}

template <typename T, typename U>
auto data_set<T, U>::label_mapper::get_mapped_value_by_label(const label_type &label) const -> const real_type & {
    if (!detail::contains(label_to_mapped_, label)) {
        throw data_set_exception{ fmt::format("Label \"{}\" unknown in this label mapping!", label) };
    }
    return label_to_mapped_.at(label);
}

template <typename T, typename U>
auto data_set<T, U>::label_mapper::get_label_by_mapped_value(const real_type &mapped_value) const -> const label_type & {
    if (!detail::contains(mapped_to_label_, mapped_value)) {
        throw data_set_exception{ fmt::format("Mapped value \"{}\" unknown in this label mapping!", mapped_value) };
    }
    return mapped_to_label_.at(mapped_value);
}

template <typename T, typename U>
auto data_set<T, U>::label_mapper::num_mappings() const noexcept -> size_type {
    PLSSVM_ASSERT(label_to_mapped_.size() == 2 && mapped_to_label_.size() == 2, "Both maps must contain exactly two values, but {} and {} were given!", label_to_mapped_.size(), mapped_to_label_.size());
    return label_to_mapped_.size();
}

template <typename T, typename U>
auto data_set<T, U>::label_mapper::labels() const -> std::vector<label_type> {
    std::vector<label_type> available_labels;
    available_labels.reserve(this->num_mappings());
    for (const auto &[key, value] : label_to_mapped_) {
        available_labels.push_back(key);
    }
    return available_labels;
}

////////////////////////////////////////////////////////////////////////////////
////                             data_set class                             ////
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
data_set<T, U>::data_set() :
    X_ptr_{ std::make_shared<std::vector<std::vector<real_type>>>() } {}

template <typename T, typename U>
data_set<T, U>::data_set(const std::string &filename) {
    // read data set from file
    this->read_file(filename, detail::ends_with(filename, ".arff") ? file_format_type::arff : file_format_type::libsvm);
}

template <typename T, typename U>
data_set<T, U>::data_set(const std::string &filename, const file_format_type format) {
    // read data set from file
    this->read_file(filename, format);
}

template <typename T, typename U>
data_set<T, U>::data_set(const std::string &filename, scaling scale_parameter) :
    data_set{ filename } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename T, typename U>
data_set<T, U>::data_set(const std::string &filename, file_format_type format, scaling scale_parameter) :
    data_set{ filename, format } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename T, typename U>
data_set<T, U>::data_set(std::vector<std::vector<real_type>> &&X) :
    X_ptr_{ std::make_shared<std::vector<std::vector<real_type>>>(std::move(X)) } {
    if (X_ptr_->empty()) {
        throw exception("Data vector is empty!");
    } else if (!std::all_of(X_ptr_->begin(), X_ptr_->end(), [&](const std::vector<real_type> &point) { return point.size() == X_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (X_ptr_->front().size() == 0) {
        throw exception{ "No features provided for the data points!" };
    }

    num_data_points_ = X_ptr_->size();
    num_features_ = X_ptr_->front().size();
}

template <typename T, typename U>
data_set<T, U>::data_set(std::vector<std::vector<real_type>> &&X, std::vector<label_type> &&y) :
    X_ptr_{ std::make_shared<std::vector<std::vector<real_type>>>(std::move(X)) }, labels_ptr_{ std::make_shared<std::vector<label_type>>(std::move(y)) } {
    if (X_ptr_->empty()) {
        throw exception("Data vector is empty!");
    } else if (!std::all_of(X_ptr_->begin(), X_ptr_->end(), [&](const std::vector<real_type> &point) { return point.size() == X_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (X_ptr_->front().size() == 0) {
        throw exception{ "No features provided for the data points!" };
    } else if (X_ptr_->size() != labels_ptr_->size()) {
        throw exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", labels_ptr_->size(), X_ptr_->size()) };
    }

    // create mapping from labels
    this->create_mapping();

    num_data_points_ = X_ptr_->size();
    num_features_ = X_ptr_->front().size();
}

template <typename T, typename U>
data_set<T, U>::data_set(std::vector<std::vector<real_type>> &&X, scaling scale_parameter) :
    data_set{ std::move(X) } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename T, typename U>
data_set<T, U>::data_set(std::vector<std::vector<real_type>> &&X, std::vector<label_type> &&y, scaling scale_parameter) :
    data_set{ std::move(X), std::move(y) } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

/******************************************************************************
 *                                Save Data Set                               *
 ******************************************************************************/
template <typename T, typename U>
void data_set<T, U>::save(const std::string &filename, const file_format_type format) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // save the data set
    switch (format) {
        case file_format_type::libsvm:
            this->write_libsvm_file(filename);
            break;
        case file_format_type::arff:
            this->write_arff_file(filename);
            break;
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Write {} data points with {} features in {} to the {} file '{}'.\n",
                   num_data_points_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   format,
                   filename);
    }
}

/******************************************************************************
 *                                   Getter                                   *
 ******************************************************************************/
template <typename T, typename U>
auto data_set<T, U>::mapped_labels() const noexcept -> optional_ref<const std::vector<real_type>> {
    if (this->has_labels()) {
        return std::make_optional(std::cref(*y_ptr_));
    } else {
        return std::nullopt;
    }
}

template <typename T, typename U>
auto data_set<T, U>::labels() const noexcept -> optional_ref<const std::vector<label_type>> {
    if (this->has_labels()) {
        return std::make_optional(std::cref(*labels_ptr_));
    } else {
        return std::nullopt;
    }
}

template <typename T, typename U>
auto data_set<T, U>::mapping() const noexcept -> optional_ref<const label_mapper> {
    if (this->has_labels()) {
        return std::make_optional(std::cref(*mapping_));
    } else {
        return std::nullopt;
    }
}

template <typename T, typename U>
auto data_set<T, U>::scaling_factors() const noexcept -> optional_ref<const scaling> {
    if (this->is_scaled()) {
        return std::make_optional(std::cref(*scale_parameters_));
    } else {
        return std::nullopt;
    }
}

template <typename T, typename U>
auto data_set<T, U>::label_from_mapped_value(const real_type val) const -> label_type {
    if (mapping_ == nullptr) {
        throw exception{ "No mapping exists!" };
    }
    return mapping_->get_label_by_mapped_value(val);
}

////////////////////////////////////////////////////////////////////////////////
////                        PRIVATE MEMBER FUNCTIONS                        ////
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
void data_set<T, U>::create_mapping() {
    PLSSVM_ASSERT(labels_ptr_ != nullptr, "Can't create mapping if no labels are provided!");

    label_mapper mapper{ *labels_ptr_ };

    // convert input labels to now mapped values
    std::vector<real_type> tmp(labels_ptr_->size());
#pragma omp parallel for default(none) shared(tmp, labels_ptr_, mapper)
    for (typename std::vector<real_type>::size_type i = 0; i < tmp.size(); ++i) {
        tmp[i] = mapper.get_mapped_value_by_label((*labels_ptr_)[i]);
    }
    y_ptr_ = std::make_shared<std::vector<real_type>>(std::move(tmp));
    mapping_ = std::make_shared<const label_mapper>(std::move(mapper));
}

template <typename T, typename U>
void data_set<T, U>::scale() {
    PLSSVM_ASSERT(this->is_scaled(), "No scaling parameters given for scaling!");

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // unpack scaling interval pair
    const real_type lower = scale_parameters_->scaling_interval.first;
    const real_type upper = scale_parameters_->scaling_interval.second;
    if (lower >= upper) {
        throw plssvm::exception(fmt::format("Illegal interval specification: lower ({}) < upper ({}).", lower, upper));
    }

    // calculate scaling factors if necessary, use provided once otherwise
    if (scale_parameters_->scaling_factors.empty()) {
        // calculate feature-wise min/max values for scaling
        for (size_type feature = 0; feature < num_features_; ++feature) {
            real_type min_value = std::numeric_limits<real_type>::max();
            real_type max_value = std::numeric_limits<real_type>::lowest();

// calculate min/max values of all data points at the specific feature
#pragma omp parallel for default(none) shared(X_ptr_) firstprivate(num_features_, feature) reduction(min                        \
                                                                                                     : min_value) reduction(max \
                                                                                                                            : max_value)
            for (size_type data_point = 0; data_point < num_data_points_; ++data_point) {
                min_value = std::min(min_value, (*X_ptr_)[data_point][feature]);
                max_value = std::max(max_value, (*X_ptr_)[data_point][feature]);
            }

            // add scaling factor only if min_value != 0.0 AND max_value != 0.0
            if (min_value != real_type{ 0.0 } && max_value != real_type{ 0.0 }) {
                scale_parameters_->scaling_factors.emplace_back(feature, min_value, max_value);
            }
        }
    } else if (scale_parameters_->scaling_factors.size() > num_features_) {
        throw invalid_file_format_exception{ fmt::format("Need at most as much scaling factors as features in the data set are present ({}), but {} were given!", num_features_, scale_parameters_->scaling_factors.size()) };
    } else if (scale_parameters_->scaling_factors.back().feature >= num_features_) {
        throw invalid_file_format_exception{ fmt::format("The maximum scaling feature index most not be greater than {}, but is {}!", scale_parameters_->scaling_factors.back().feature, num_features_) };
    }

// scale values
#pragma omp parallel for default(none) shared(scale_parameters_, X_ptr_) firstprivate(lower, upper, num_data_points_)
    for (size_type i = 0; i < scale_parameters_->scaling_factors.size(); ++i) {
        // extract feature-wise min and max values
        const typename scaling::factors factor = scale_parameters_->scaling_factors[i];
        // scale data values
        for (size_type data_point = 0; data_point < num_data_points_; ++data_point) {
            (*X_ptr_)[data_point][factor.feature] = (upper - lower) / (factor.upper - factor.lower) * ((*X_ptr_)[data_point][factor.feature] - factor.upper) + upper;
        }
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Scaled the data set to the range [{}, {}] in {}.\n", lower, upper, std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

/******************************************************************************
 *                                 Write Files                                *
 ******************************************************************************/
template <typename T, typename U>
void data_set<T, U>::write_libsvm_file(const std::string &filename) const {
    // write data
    if (this->has_labels()) {
        detail::io::write_libsvm_data(filename, *X_ptr_, *labels_ptr_);
    } else {
        detail::io::write_libsvm_data(filename, *X_ptr_);
    }
}

template <typename T, typename U>
void data_set<T, U>::write_arff_file(const std::string &filename) const {
    // write file
    if (this->has_labels()) {
        detail::io::write_arff_data(filename, *X_ptr_, *labels_ptr_);
    } else {
        detail::io::write_arff_data(filename, *X_ptr_);
    }
}

/******************************************************************************
 *                                 Read Files                                 *
 ******************************************************************************/
template <typename T, typename U>
void data_set<T, U>::read_file(const std::string &filename, file_format_type format) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // parse the given file
    switch (format) {
        case file_format_type::libsvm:
            read_libsvm_file(filename);
            break;
        case file_format_type::arff:
            read_arff_file(filename);
            break;
    }

    // create label mapping
    if (this->has_labels()) {
        this->create_mapping();
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Read {} data points with {} features in {} using the {} parser from file '{}'.\n\n",
                   num_data_points_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   format,
                   filename);
    }
}

// TODO: condense some functions since they are completely moved to the utility header

template <typename T, typename U>
void data_set<T, U>::read_libsvm_file(const std::string &filename) {
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    std::vector<std::vector<real_type>> data{};
    std::vector<label_type> label{};

    std::tie(num_data_points_, num_features_, data, label) = detail::io::parse_libsvm_data<real_type, label_type>(reader);

    // update shared pointer
    X_ptr_ = std::make_shared<decltype(data)>(std::move(data));
    if (label.empty()) {
        labels_ptr_ = nullptr;
    } else {
        labels_ptr_ = std::make_shared<decltype(label)>(std::move(label));
    }
}

template <typename T, typename U>
void data_set<T, U>::read_arff_file(const std::string &filename) {
    detail::io::file_reader reader{ filename };
    reader.read_lines('%');

    std::vector<std::vector<real_type>> data{};
    std::vector<label_type> label{};

    // parse file
    std::tie(num_data_points_, num_features_, data, label) = detail::io::parse_arff_data<real_type, label_type>(reader);

    // update shared pointer
    X_ptr_ = std::make_shared<decltype(data)>(std::move(data));
    if (label.empty()) {
        labels_ptr_ = nullptr;
    } else {
        labels_ptr_ = std::make_shared<decltype(label)>(std::move(label));
    }
}

namespace detail {

// two possible types: real_type + int and real_type + std::string
using data_set_variants = std::variant<plssvm::data_set<float>, plssvm::data_set<float, std::string>, plssvm::data_set<double>, plssvm::data_set<double, std::string>>;

template <typename real_type, typename label_type = typename data_set<real_type>::label_type>
inline data_set_variants data_set_factory_impl(const cmd::parameter_train &params) {
    return data_set_variants{ plssvm::data_set<real_type, label_type>{ params.input_filename } };
}
template <typename real_type, typename label_type = typename data_set<real_type>::label_type>
inline data_set_variants data_set_factory_impl(const cmd::parameter_predict &params) {
    return data_set_variants{ plssvm::data_set<real_type, label_type>{ params.input_filename } };
}
template <typename real_type, typename label_type = typename data_set<real_type>::label_type>
inline data_set_variants data_set_factory_impl(const cmd::parameter_scale &params) {
    if (!params.restore_filename.empty()) {
        return data_set_variants{ plssvm::data_set<real_type, label_type>{ params.input_filename, { params.restore_filename } } };
    } else {
        return data_set_variants{ plssvm::data_set<real_type, label_type>{ params.input_filename, { static_cast<real_type>(params.lower), static_cast<real_type>(params.upper) } } };
    }
}

template <typename cmd_parameter>
inline data_set_variants data_set_factory(const cmd_parameter &params) {
    if (params.float_as_real_type && params.strings_as_labels) {
        return data_set_factory_impl<float, std::string>(params);
    } else if (params.float_as_real_type && !params.strings_as_labels) {
        return data_set_factory_impl<float>(params);
    } else if (!params.float_as_real_type && params.strings_as_labels) {
        return data_set_factory_impl<double, std::string>(params);
    } else {
        return data_set_factory_impl<double>(params);
    }
}

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_HPP_
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a data set class encapsulating all data points, features, and potential labels for the classification task.
 */

#ifndef PLSSVM_DATA_SET_CLASSIFICATION_DATA_SET_HPP_
#define PLSSVM_DATA_SET_CLASSIFICATION_DATA_SET_HPP_
#pragma once

#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/data_set/data_set.hpp"                    // plssvm::data_set
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/type_list.hpp"                     // plssvm::detail::{supported_label_types_classification, tuple_contains_v}
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::data_set_exception
#include "plssvm/file_format_types.hpp"                    // plssvm::file_format_type
#include "plssvm/matrix.hpp"                               // plssvm::aos_matrix
#include "plssvm/shape.hpp"                                // plssvm::shape
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include <chrono>    // std::chrono::{time_point, steady_clock, duration_cast, millisecond}
#include <cstddef>   // std::size_t
#include <map>       // std::map
#include <memory>    // std::shared_ptr, std::make_shared
#include <optional>  // std::optional, std::make_optional, std::nullopt
#include <set>       // std::set
#include <string>    // std::string
#include <utility>   // std::forward, std::move
#include <vector>    // std::vector

namespace plssvm {

// forward declare C-SVC class
class csvc;

/**
 * @brief Encapsulate all necessary data that is needed for training or predicting using an C-SVC.
 * @details May or may not contain labels!
 *          Internally, saves all data using [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr) to make a plssvm::classification_data_set relatively cheap to copy!
 * @tparam U the label type of the data (must be an arithmetic type or `std::string`; default: `int`)
 */
template <typename U = int>
class classification_data_set : public data_set<U> {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<U, detail::supported_label_types_classification>,
                  "Illegal label type for classification provided! See the 'supported_label_types_classification' in the type_list.hpp header for a list of the allowed types.");

    // befriend C-SVC class used with the classification data set
    friend class csvc;
    // befriend C-SVC model used with the classification data set
    template <typename>
    friend class classification_model;

    /// The base data set class.
    using base_data_set = data_set<U>;

    using base_data_set::data_ptr_;
    using base_data_set::labels_ptr_;
    using base_data_set::num_data_points_;
    using base_data_set::num_features_;
    using base_data_set::y_ptr_;

    // forward declare the label_mapper class
    class label_mapper;

  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using typename base_data_set::label_type;
    /// An unsigned integer type.
    using typename base_data_set::size_type;
    /// The C-SVM type used with this data set.
    using svm_fit_type = ::plssvm::csvc;

    /**
     * @brief Construct a new classification data set by forwarding all provided arguments to the data_set constructors.
     * @tparam Args the type of the provided arguments
     * @param[in] args the provided arguments forwarded to the base class constructors
     */
    template <typename... Args>
    explicit classification_data_set(Args &&...args);

    /**
     * @copydoc plssvm::data_set::save
     */
    void save(const std::string &filename, file_format_type format) const override;

    /**
     * @brief Returns an optional to the classes in this data set.
     * @details If the data set contains the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns the labels `{ -1, 1 }`.
     * @note Must not return a optional reference, since it would bind to a temporary!
     * @return if this data set contains labels, returns a reference to all classes, otherwise returns a `std::nullopt` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::optional<std::vector<label_type>> classes() const;

    /**
     * @brief Returns the number of classes in this data set.
     * @details If the data set contains the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns `2`.
     *          It is the same as: `dataset.classes()->size()`
     * @return the number of classes (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_classes() const noexcept { return mapping_ != nullptr ? mapping_->num_mappings() : 0; }

  private:
    /**
     * @copydoc plssvm::data_set::map_label
     */
    void map_label() override;

    /// The mapping used to convert the original label to its mapped value and vice versa; may be `nullptr` if no labels have been provided.
    std::shared_ptr<const label_mapper> mapping_{ nullptr };
};

//*************************************************************************************************************************************//
//                                                      label mapper nested-class                                                      //
//*************************************************************************************************************************************//

/**
 * @brief Implements all necessary functionality to map arbitrary labels to labels usable by the C-SVMs.
 * @details Currently maps all labels to { -1 , 1 }.
 */
template <typename U>
class classification_data_set<U>::label_mapper {
  public:
    /**
     * @brief Create a mapping from all labels to their index used in the right-hand side when solving the system of linear equations and vice versa.
     * @param[in] classes the labels to map
     * @note Currently only binary classification is supported, i.e., only two different labels may be provided!
     * @throws plssvm::data_set_exception if not exactly two different labels are provided
     */
    explicit label_mapper(const std::vector<label_type> &classes);

    /**
     * @brief Given the original label value, return the mapped index in the one vs. all mapping.
     * @param[in] label the original label value
     * @throws plssvm::data_set_exception if the original label value does not exist in this mapping
     * @return the mapped index (`[[nodiscard]]`)
     */
    [[nodiscard]] const size_type &get_mapped_index_by_label(const label_type &label) const;
    /**
     * @brief Given the mapped index in the one vs. all mapping, return the original label value.
     * @param[in] mapped_index the mapped index
     * @throws plssvm::data_set_exception if the mapped index does not exist in this mapping
     * @return the original label value (`[[nodiscard]]`)
     */
    [[nodiscard]] const label_type &get_label_by_mapped_index(const size_type &mapped_index) const;
    /**
     * @brief Returns the number of valid mappings. This is equivalent to the number of different labels.
     * @return the number of valid mapping entries (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_mappings() const noexcept;
    /**
     * @brief Return a vector containing the different, original labels of the current data set.
     * @return the original labels (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<label_type> labels() const;

  private:
    /// A mapping from the label to its mapped index in the right-hand side vector.
    std::map<label_type, size_type> label_to_index_{};
    /// A mapping from the mapped index to the original label value.
    std::map<size_type, label_type> index_to_label_{};
};

/// @cond Doxygen_suppress
template <typename U>
classification_data_set<U>::classification_data_set::label_mapper::label_mapper(const std::vector<label_type> &classes) {
    PLSSVM_ASSERT(std::set(classes.cbegin(), classes.cend()).size() == classes.size(),
                  "The provided labels for the label_mapper must not include duplicated ones!");
    // create mapping
    for (std::size_t idx = 0; idx < classes.size(); ++idx) {
        label_to_index_[classes[idx]] = idx;
        index_to_label_[idx] = classes[idx];
    }
}

/// @endcond

template <typename U>
auto classification_data_set<U>::label_mapper::get_mapped_index_by_label(const label_type &label) const -> const size_type & {
    if (!detail::contains(label_to_index_, label)) {
        throw data_set_exception{ fmt::format("Label \"{}\" unknown in this label mapping!", label) };
    }
    return label_to_index_.at(label);
}

template <typename U>
auto classification_data_set<U>::label_mapper::get_label_by_mapped_index(const size_type &mapped_index) const -> const label_type & {
    if (!detail::contains(index_to_label_, mapped_index)) {
        throw data_set_exception{ fmt::format("Mapped index \"{}\" unknown in this label mapping!", mapped_index) };
    }
    return index_to_label_.at(mapped_index);
}

template <typename U>
auto classification_data_set<U>::label_mapper::num_mappings() const noexcept -> size_type {
    PLSSVM_ASSERT(label_to_index_.size() == index_to_label_.size(), "Both maps must contain the same number of values, but {} and {} were given!", label_to_index_.size(), index_to_label_.size());
    return label_to_index_.size();
}

template <typename U>
auto classification_data_set<U>::label_mapper::labels() const -> std::vector<label_type> {
    std::vector<label_type> available_labels;
    available_labels.reserve(this->num_mappings());
    for (const auto &[key, value] : label_to_index_) {
        available_labels.push_back(key);
    }
    return available_labels;
}

//*************************************************************************************************************************************//
//                                                    classification data set class                                                    //
//*************************************************************************************************************************************//

template <typename U>
template <typename... Args>
classification_data_set<U>::classification_data_set(Args &&...args) :
    base_data_set{ std::forward<Args>(args)... } {
    // create label mapping
    if (this->has_labels()) {
        this->map_label();
    }

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "data_set_create", "type", "classification" }));
    if (this->has_labels()) {
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Created a classification data set with {} data points, {} features, and {} classes.\n",
                    detail::tracking::tracking_entry{ "data_set_create", "num_data_points", num_data_points_ },
                    detail::tracking::tracking_entry{ "data_set_create", "num_features", num_features_ },
                    detail::tracking::tracking_entry{ "data_set_create", "num_classes", this->num_classes() });
    } else {
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Created a classification data set with {} data points and {} features.\n",
                    detail::tracking::tracking_entry{ "data_set_create", "num_data_points", num_data_points_ },
                    detail::tracking::tracking_entry{ "data_set_create", "num_features", num_features_ });
    }
}

template <typename U>
void classification_data_set<U>::save(const std::string &filename, const file_format_type format) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // save the data set
    base_data_set::save(filename, format);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Write {} classification data points with {} features and {} classes in {} to the {} file '{}'.\n",
                detail::tracking::tracking_entry{ "data_set_write", "num_data_points", num_data_points_ },
                detail::tracking::tracking_entry{ "data_set_write", "num_features", num_features_ },
                detail::tracking::tracking_entry{ "data_set_write", "num_classes", this->num_classes() },
                detail::tracking::tracking_entry{ "data_set_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "data_set_write", "format", format },
                detail::tracking::tracking_entry{ "data_set_write", "filename", filename });
}

template <typename U>
auto classification_data_set<U>::classes() const -> std::optional<std::vector<label_type>> {
    if (this->has_labels()) {
        return std::make_optional(mapping_->labels());
    }
    return std::nullopt;
}

//*************************************************************************************************************************************//
//                                                      PRIVATE MEMBER FUNCTIONS                                                       //
//*************************************************************************************************************************************//

template <typename U>
void classification_data_set<U>::map_label() {
    PLSSVM_ASSERT(labels_ptr_ != nullptr, "Can't create mapping if no labels are provided!");

    // create vector containing unique classes
    std::set<label_type> unique_labels(labels_ptr_->cbegin(), labels_ptr_->cend());
    std::vector<label_type> classes(unique_labels.cbegin(), unique_labels.cend());

    // create label mapping
    label_mapper mapper{ classes };

    // convert input labels to now mapped values
    aos_matrix<real_type> tmp{ shape{ mapper.num_mappings(), labels_ptr_->size() }, real_type{ -1.0 } };

#pragma omp parallel for collapse(2)
    for (typename std::vector<std::vector<real_type>>::size_type label = 0; label < tmp.num_rows(); ++label) {
        for (typename std::vector<real_type>::size_type i = 0; i < tmp.num_cols(); ++i) {
            if (label == mapper.get_mapped_index_by_label((*labels_ptr_)[i])) {
                tmp(label, i) = real_type{ 1.0 };
            }
        }
    }

    y_ptr_ = std::make_shared<decltype(tmp)>(std::move(tmp));
    mapping_ = std::make_shared<const label_mapper>(std::move(mapper));
}

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_CLASSIFICATION_DATA_SET_HPP_

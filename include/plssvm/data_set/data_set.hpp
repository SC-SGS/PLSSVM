/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a data set class encapsulating all data points, features, and potential labels.
 */

#ifndef PLSSVM_DATA_SET_DATA_SET_HPP_
#define PLSSVM_DATA_SET_DATA_SET_HPP_
#pragma once

#include "plssvm/constants.hpp"                            // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set/min_max_scaler.hpp"              // plssvm::min_max_scaler
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/io/arff_parsing.hpp"               // plssvm::detail::io::write_libsvm_data
#include "plssvm/detail/io/file_reader.hpp"                // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_parsing.hpp"             // plssvm::detail::io::write_arff_data
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/string_utility.hpp"                // plssvm::detail::ends_with
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::data_set_exception
#include "plssvm/file_format_types.hpp"                    // plssvm::file_format_type
#include "plssvm/matrix.hpp"                               // plssvm::soa_matrix
#include "plssvm/shape.hpp"                                // plssvm::shape
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format

#include <algorithm>   // std::max, std::min, std::sort, std::adjacent_find
#include <chrono>      // std::chrono::{time_point, steady_clock, duration_cast, millisecond}
#include <cstddef>     // std::size_t
#include <functional>  // std::reference_wrapper, std::cref
#include <limits>      // std::numeric_limits::{max, lowest}
#include <memory>      // std::shared_ptr, std::make_shared
#include <optional>    // std::optional, std::make_optional, std::nullopt
#include <string>      // std::string
#include <tuple>       // std::tie
#include <utility>     // std::move, std::pair, std::make_pair
#include <vector>      // std::vector

namespace plssvm {

/**
 * @brief Type alias for an optional reference (since `std::optional<T&>` is not allowed).
 * @tparam T the type to wrap as a reference
 */
template <typename T>
using optional_ref = std::optional<std::reference_wrapper<T>>;

/**
 * @brief Encapsulate all necessary data that is needed for training or predicting using an SVM.
 * @details May or may not contain labels!
 *          Internally, saves all data using [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr) to make a plssvm::data_set relatively cheap to copy!
 * @tparam U the label type of the data (must be an arithmetic type or `std::string`; default: `int`)
 */
template <typename U>
class data_set {
  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using label_type = U;
    /// An unsigned integer type.
    using size_type = std::size_t;

    /**
     * @brief Read the data points from the file @p filename.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] filename the file to read the data points from
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     */
    explicit data_set(const std::string &filename);
    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the @p plssvm::file_format_type.
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     */
    data_set(const std::string &filename, file_format_type format);
    /**
     * @brief Read the data points from the file @p filename and scale it using the provided @p scaler.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] filename the file to read the data points from
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    data_set(const std::string &filename, min_max_scaler scaler);
    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the plssvm::file_format_type @p format and
     *        scale it using the provided @p scaler.
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    data_set(const std::string &filename, file_format_type format, min_max_scaler scaler);

    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    explicit data_set(const std::vector<std::vector<real_type>> &data_points);
    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix and copying the @p labels.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels);
    /**
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and scale them using the provided @p scaler.
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    data_set(const std::vector<std::vector<real_type>> &data_points, min_max_scaler scaler);
    /**
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and copying the @p labels and scale the @p data_points using the provided @p scaler.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, min_max_scaler scaler);

    /**
     * @brief Create a new data set from the provided @p data_points.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    template <layout_type layout>
    explicit data_set(const matrix<real_type, layout> &data_points);
    /**
     * @brief Create a new data set from the provided @p data_points and @p labels.
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    template <layout_type layout>
    data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels);
    /**
     * @brief Create a new data set from the the provided @p data_points and scale them using the provided @p scaler.
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    template <layout_type layout>
    data_set(const matrix<real_type, layout> &data_points, min_max_scaler scaler);
    /**
     * @brief Create a new data set from the the provided @p data_points and @p labels and scale the @p data_points using the provided @p scaler.
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    template <layout_type layout>
    data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels, min_max_scaler scaler);

    /**
     * @brief Use the provided @p data_points in this data set.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @note Moves the @p data_points into this data set. If @p data_points have the wrong padding, a runtime exception is thrown.
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the padding sizes of @p data_points are wrong
     */
    explicit data_set(soa_matrix<real_type> &&data_points);
    /**
     * @brief Use the provided @p data_points and @p labels in this data set.
     * @note Moves the @p data_points and @p labels into this data set. If @p data_points have the wrong padding, a runtime exception is thrown.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the padding sizes of @p data_points are wrong
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    data_set(soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels);
    /**
     * @brief Use the provided @p data_points in this data set and scale them using the provided @p scaler.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @note Moves the @p data_points into this data set. If @p data_points have the wrong padding, a runtime exception is thrown.
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the padding sizes of @p data_points are wrong
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    data_set(soa_matrix<real_type> &&data_points, min_max_scaler scaler);
    /**
     * @brief Use the provided @p data_points and @p labels in this data set and scale them using the provided @p scaler.
     * @note Moves the @p data_points and @p labels into this data set. If @p data_points have the wrong padding, a runtime exception is thrown.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the padding sizes of @p data_points are wrong
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    data_set(soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels, min_max_scaler scaler);

    /**
     * @brief Default copy constructor.
     */
    data_set(const data_set &) = default;
    /**
     * @brief Default move constructor.
     */
    data_set(data_set &&) noexcept = default;
    /**
     * @brief Default copy assignment operator.
     * @return `*this`
     */
    data_set &operator=(const data_set &) = default;
    /**
     * @brief Default move assignment operator.
     * @return `*this`
     */
    data_set &operator=(data_set &&) noexcept = default;

    /**
     * @brief Virtual destructor to allow derived classes to clean up properly.
     */
    virtual ~data_set() = default;

    /**
     * @brief Save the data points and potential labels of this data set to the file @p filename using the file @p format type.
     * @param[in] filename the file to save the data points and labels to
     * @param[in] format the file format
     */
    virtual void save(const std::string &filename, file_format_type format) const;
    /**
     * @brief Save the data points and potential labels of this data set to the file @p filename.
     * @details Automatically determines the plssvm::file_format_type based on the file extension.
     *          If the file extension isn't `.arff`, saves the data as `.libsvm` file.
     * @param[in] filename the file to save the data points and labels to
     */
    void save(const std::string &filename) const;

    /**
     * @brief Return the data points in this data set by copying them to a 2D vector.
     * @return the data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const soa_matrix<real_type> &data() const { return *data_ptr_; }

    /**
     * @brief Returns whether this data set contains labels or not.
     * @return `true` if this data set contains labels, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool has_labels() const noexcept { return labels_ptr_ != nullptr; }

    /**
     * @brief Returns an optional reference to the labels in this data set.
     * @details If the labels are present, they can be retrieved as `std::vector` using: `dataset.labels()->%get()`.
     * @return if this data set contains labels, returns a reference to them, otherwise returns a `std::nullopt` (`[[nodiscard]]`)
     */
    [[nodiscard]] optional_ref<const std::vector<label_type>> labels() const noexcept;

    /**
     * @brief Returns the number of data points in this data set.
     * @return the number of data points (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_data_points() const noexcept { return num_data_points_; }

    /**
     * @brief Returns the number of features in this data set.
     * @return the number of features (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_features() const noexcept { return num_features_; }

    /**
     * @brief Returns whether this data set has been scaled or not.
     * @details The used scaling factors can be retrieved using plssvm::data_set::scaling_factors().
     * @return `true` if this data set has been scaled, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_scaled() const noexcept { return scaler_ != nullptr; }

    /**
     * @brief Returns the scaling factors as an optional reference used to scale the data points in this data set.
     * @details Can be used to scale another data set in the same way (e.g., a test data set).
     *          If the data set has been scaled, the scaling factors can be retrieved as using: `dataset.scaling_factors()->%get()`.
     * @return the scaling factors (`[[nodiscard]]`)
     */
    [[nodiscard]] optional_ref<const min_max_scaler> scaling_factors() const noexcept;

  protected:
    /**
     * @brief Default construct an empty data set.
     */
    data_set() :
        data_ptr_{ std::make_shared<soa_matrix<real_type>>() } { }

    /**
     * @brief Create the mapping between the provided labels and the internally used values.
     * @throws plssvm::data_set_exception any exception of the plssvm::data_set::label_mapper class
     */
    virtual void map_label() = 0;

    /**
     * @brief Read the data points and potential labels from the file @p filename assuming the plssvm::file_format_type @p format.
     * @param[in] filename the filename to read the data from
     * @param[in] format the assumed file format type
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by the respective functions in the plssvm::detail::io namespace
     * @throws plssvm::data_set_exception if labels are present in @p filename, all exceptions thrown by plssvm::data_set::map_label
     */
    void read_file(const std::string &filename, file_format_type format);

    /// The number of data points in this data set.
    size_type num_data_points_{ 0 };
    /// The number of features in this data set.
    size_type num_features_{ 0 };

    /// A pointer to the two-dimensional data points.
    std::shared_ptr<soa_matrix<real_type>> data_ptr_{ nullptr };
    /// A pointer to the original labels of this data set; may be `nullptr` if no labels have been provided.
    std::shared_ptr<std::vector<label_type>> labels_ptr_{ nullptr };
    /// A pointer to the mapped values of the labels of this data set; may be `nullptr` if no labels have been provided.
    std::shared_ptr<aos_matrix<real_type>> y_ptr_{ nullptr };

    /// The min-max scaling parameters used to scale the data points in this data set; may be `nullptr` if no data point scaling was requested.
    std::shared_ptr<min_max_scaler> scaler_{ nullptr };
};

//*************************************************************************************************************************************//
//                                                           data set class                                                            //
//*************************************************************************************************************************************//

template <typename U>
data_set<U>::data_set(const std::string &filename) {
    // read data set from file
    // if the file doesn't end with .arff, assume a LIBSVM file
    this->read_file(filename, detail::ends_with(filename, ".arff") ? file_format_type::arff : file_format_type::libsvm);
}

template <typename U>
data_set<U>::data_set(const std::string &filename, const file_format_type format) {
    // read data set from file
    this->read_file(filename, format);
}

template <typename U>
data_set<U>::data_set(const std::string &filename, min_max_scaler scale_parameter) :
    data_set{ filename } {
    // initialize scaling
    scaler_ = std::make_shared<min_max_scaler>(std::move(scale_parameter));
    // scale data set
    scaler_->scale(*data_ptr_);
}

template <typename U>
data_set<U>::data_set(const std::string &filename, file_format_type format, min_max_scaler scale_parameter) :
    data_set{ filename, format } {
    // initialize scaling
    scaler_ = std::make_shared<min_max_scaler>(std::move(scale_parameter));
    // scale data set
    scaler_->scale(*data_ptr_);
}

// clang-format off
template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points) try :
    data_set{ soa_matrix<real_type>{ data_points, shape{ PADDING_SIZE, PADDING_SIZE } } } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels) try :
    data_set{ soa_matrix<real_type>{ data_points, shape{ PADDING_SIZE, PADDING_SIZE } }, std::move(labels) } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points, min_max_scaler scale_parameter) try :
    data_set{ soa_matrix<real_type>{ data_points, shape{ PADDING_SIZE, PADDING_SIZE } }, std::move(scale_parameter) } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, min_max_scaler scale_parameter) try :
    data_set{ soa_matrix<real_type>{ data_points, shape{ PADDING_SIZE, PADDING_SIZE } }, std::move(labels), std::move(scale_parameter) } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

// clang-format on

template <typename U>
template <layout_type layout>
data_set<U>::data_set(const matrix<real_type, layout> &data_points) :
    num_data_points_{ data_points.num_rows() },
    num_features_{ data_points.num_cols() },
    data_ptr_{ std::make_shared<soa_matrix<real_type>>(data_points, shape{ PADDING_SIZE, PADDING_SIZE }) } {
    // the provided data points vector may not be empty
    if (data_ptr_->num_rows() == 0) {
        throw data_set_exception{ "Data vector is empty!" };
    }
    if (data_ptr_->num_cols() == 0) {
        throw data_set_exception{ "No features provided for the data points!" };
    }
}

template <typename U>
template <layout_type layout>
data_set<U>::data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels) :
    num_data_points_{ data_points.num_rows() },
    num_features_{ data_points.num_cols() },
    data_ptr_{ std::make_shared<soa_matrix<real_type>>(data_points, shape{ PADDING_SIZE, PADDING_SIZE }) },
    labels_ptr_{ std::make_shared<std::vector<label_type>>(std::move(labels)) } {
    // the provided data points vector may not be empty
    if (data_ptr_->num_rows() == 0) {
        throw data_set_exception{ "Data vector is empty!" };
    }
    if (data_ptr_->num_cols() == 0) {
        throw data_set_exception{ "No features provided for the data points!" };
    }
    // the number of labels must be equal to the number of data points!
    if (data_ptr_->num_rows() != labels_ptr_->size()) {
        throw data_set_exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", labels_ptr_->size(), data_ptr_->num_rows()) };
    }
}

template <typename U>
template <layout_type layout>
data_set<U>::data_set(const matrix<real_type, layout> &data_points, min_max_scaler scale_parameter) :
    data_set{ data_points } {
    // initialize scaling
    scaler_ = std::make_shared<min_max_scaler>(std::move(scale_parameter));
    // scale data set
    scaler_->scale(*data_ptr_);
}

template <typename U>
template <layout_type layout>
data_set<U>::data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels, min_max_scaler scale_parameter) :
    data_set{ data_points, std::move(labels) } {
    // initialize scaling
    scaler_ = std::make_shared<min_max_scaler>(std::move(scale_parameter));
    // scale data set
    scaler_->scale(*data_ptr_);
}

template <typename U>
data_set<U>::data_set(soa_matrix<real_type> &&data_points) :
    num_data_points_{ data_points.num_rows() },
    num_features_{ data_points.num_cols() },
    data_ptr_{ std::make_shared<soa_matrix<real_type>>(std::move(data_points)) } {
    // the provided data points vector may not be empty
    if (data_ptr_->num_rows() == 0) {
        throw data_set_exception{ "Data vector is empty!" };
    }
    if (data_ptr_->num_cols() == 0) {
        throw data_set_exception{ "No features provided for the data points!" };
    }
    // the padding must be correct
    if (data_ptr_->padding() != shape{ PADDING_SIZE, PADDING_SIZE }) {
        throw data_set_exception{ fmt::format("Data vector has the wring padding ({})!", data_ptr_->padding()) };
    }
}

template <typename U>
data_set<U>::data_set(soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels) :
    num_data_points_{ data_points.num_rows() },
    num_features_{ data_points.num_cols() },
    data_ptr_{ std::make_shared<soa_matrix<real_type>>(std::move(data_points)) },
    labels_ptr_{ std::make_shared<std::vector<label_type>>(std::move(labels)) } {
    // the provided data points vector may not be empty
    if (data_ptr_->num_rows() == 0) {
        throw data_set_exception{ "Data vector is empty!" };
    }
    if (data_ptr_->num_cols() == 0) {
        throw data_set_exception{ "No features provided for the data points!" };
    }
    // the number of labels must be equal to the number of data points!
    if (data_ptr_->num_rows() != labels_ptr_->size()) {
        throw data_set_exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", labels_ptr_->size(), data_ptr_->num_rows()) };
    }
    // the padding must be correct
    if (data_ptr_->padding() != shape{ PADDING_SIZE, PADDING_SIZE }) {
        throw data_set_exception{ fmt::format("Data vector has the wring padding ({})!", data_ptr_->padding()) };
    }
}

template <typename U>
data_set<U>::data_set(soa_matrix<real_type> &&data_points, min_max_scaler scale_parameter) :
    data_set{ std::move(data_points) } {
    // initialize scaling
    scaler_ = std::make_shared<min_max_scaler>(std::move(scale_parameter));
    // scale data set
    scaler_->scale(*data_ptr_);
}

template <typename U>
data_set<U>::data_set(soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels, min_max_scaler scale_parameter) :
    data_set{ std::move(data_points), std::move(labels) } {
    // initialize scaling
    scaler_ = std::make_shared<min_max_scaler>(std::move(scale_parameter));
    // scale data set
    scaler_->scale(*data_ptr_);
}

template <typename U>
void data_set<U>::save(const std::string &filename, const file_format_type format) const {
    // save the data set
    if (this->has_labels()) {
        // save data with labels
        switch (format) {
            case file_format_type::libsvm:
                detail::io::write_libsvm_data(filename, *data_ptr_, *labels_ptr_);
                break;
            case file_format_type::arff:
                detail::io::write_arff_data(filename, *data_ptr_, *labels_ptr_);
                break;
        }
    } else {
        // save data without labels
        switch (format) {
            case file_format_type::libsvm:
                detail::io::write_libsvm_data(filename, *data_ptr_);
                break;
            case file_format_type::arff:
                detail::io::write_arff_data(filename, *data_ptr_);
                break;
        }
    }
}

template <typename U>
void data_set<U>::save(const std::string &filename) const {
    if (detail::ends_with(filename, ".arff")) {
        this->save(filename, file_format_type::arff);
    } else {
        this->save(filename, file_format_type::libsvm);
    }
}

template <typename U>
auto data_set<U>::labels() const noexcept -> optional_ref<const std::vector<label_type>> {
    if (this->has_labels()) {
        return std::make_optional(std::cref(*labels_ptr_));
    }
    return std::nullopt;
}

template <typename U>
auto data_set<U>::scaling_factors() const noexcept -> optional_ref<const min_max_scaler> {
    if (this->is_scaled()) {
        return std::make_optional(std::cref(*scaler_));
    }
    return std::nullopt;
}

//*************************************************************************************************************************************//
//                                                      PRIVATE MEMBER FUNCTIONS                                                       //
//*************************************************************************************************************************************//

template <typename U>
void data_set<U>::read_file(const std::string &filename, file_format_type format) {
    // get the comment character based on the file_format_type
    char comment{ ' ' };
    switch (format) {
        case file_format_type::libsvm:
            comment = '#';
            break;
        case file_format_type::arff:
            comment = '%';
            break;
    }

    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // create the empty placeholders
    typename decltype(data_ptr_)::element_type data{};
    typename decltype(labels_ptr_)::element_type label{};

    // parse the given file
    switch (format) {
        case file_format_type::libsvm:
            std::tie(num_data_points_, num_features_, data, label) = detail::io::parse_libsvm_data<label_type>(reader);
            break;
        case file_format_type::arff:
            std::tie(num_data_points_, num_features_, data, label) = detail::io::parse_arff_data<label_type>(reader);
            break;
    }

    // update shared pointer
    data_ptr_ = std::make_shared<decltype(data)>(std::move(data));
    if (label.empty()) {
        labels_ptr_ = nullptr;
    } else {
        labels_ptr_ = std::make_shared<decltype(label)>(std::move(label));
    }
}

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_DATA_SET_HPP_

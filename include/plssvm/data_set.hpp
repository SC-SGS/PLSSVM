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

#ifndef PLSSVM_DATA_SET_HPP_
#define PLSSVM_DATA_SET_HPP_
#pragma once

#include "plssvm/constants.hpp"                          // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/io/arff_parsing.hpp"             // plssvm::detail::io::{read_libsvm_data, write_libsvm_data}
#include "plssvm/detail/io/file_reader.hpp"              // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_parsing.hpp"           // plssvm::detail::io::{read_arff_data, write_arff_data}
#include "plssvm/detail/io/scaling_factors_parsing.hpp"  // plssvm::detail::io::{parse_scaling_factors, read_scaling_factors}
#include "plssvm/detail/logging.hpp"                     // plssvm::detail::log
#include "plssvm/detail/performance_tracker.hpp"         // plssvm::detail::tracking_entry
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::ends_with
#include "plssvm/detail/type_list.hpp"                   // plssvm::detail::{supported_label_types, tuple_contains_v}
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"              // plssvm::data_set_exception
#include "plssvm/file_format_types.hpp"                  // plssvm::file_format_type
#include "plssvm/matrix.hpp"                             // plssvm::soa_matrix
#include "plssvm/verbosity_levels.hpp"                   // plssvm::verbosity_level

#include "fmt/chrono.h"   // directly output std::chrono times via fmt
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // directly output objects with operator<< overload via fmt

#include <algorithm>   // std::all_of, std::max, std::min, std::sort, std::adjacent_find
#include <array>       // std::array
#include <chrono>      // std::chrono::{time_point, steady_clock, duration_cast, millisecond}
#include <cstddef>     // std::size_t
#include <functional>  // std::reference_wrapper, std::cref
#include <iostream>    // std::cout, std::endl
#include <limits>      // std::numeric_limits::{max, lowest}
#include <map>         // std::map
#include <memory>      // std::shared_ptr, std::make_shared
#include <optional>    // std::optional, std::make_optional, std::nullopt
#include <set>         // std::set
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
 * @example data_set_examples.cpp
 * @brief A few examples regarding the plssvm::data_set class.
 */

/**
 * @brief Encapsulate all necessary data that is needed for training or predicting using an SVM.
 * @details May or may not contain labels!
 *          Internally, saves all data using [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr) to make a plssvm::data_set relatively cheap to copy!
 * @tparam U the label type of the data (must be an arithmetic type or `std::string`; default: `int`)
 */
template <typename U = int>
class data_set {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<U, detail::supported_label_types>, "Illegal label type provided! See the 'label_type_list' in the type_list.hpp header for a list of the allowed types.");

    // plssvm::model needs the default constructor
    template <typename>
    friend class model;
    // plssvm::csvm needs the label mapping
    friend class csvm;

  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using label_type = U;
    /// An unsigned integer type.
    using size_type = std::size_t;

    // forward declare the scaling class
    class scaling;
    // forward declare the label_mapper class
    class label_mapper;

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
     * @brief Read the data points from the file @p filename and scale it using the provided @p scale_parameter.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] filename the file to read the data points from
     * @param[in] scale_parameter the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::data_set_exception all exceptions thrown by plssvm::data_set::scale
     */
    data_set(const std::string &filename, scaling scale_parameter);
    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the plssvm::file_format_type @p format and
     *        scale it using the provided @p scale_parameter.
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @param[in] scale_parameter the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::data_set_exception all exceptions thrown by plssvm::data_set::scale
     */
    data_set(const std::string &filename, file_format_type format, scaling scale_parameter);

    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvm::fit!
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
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and scale them using the provided @p scale_parameter.
     * @param[in] data_points the data points used in this data set
     * @param[in] scale_parameter the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception all exceptions thrown by plssvm::data_set::scale
     */
    data_set(const std::vector<std::vector<real_type>> &data_points, scaling scale_parameter);
    /**
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and copying the @p labels and scale the @p data_points using the provided @p scale_parameter.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scale_parameter the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::data_set_exception all exceptions thrown by plssvm::data_set::scale
     */
    data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, scaling scale_parameter);

    /**
     * @brief Create a new data set from the provided @p data_points.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvm::fit!
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the row-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the column-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    template <layout_type layout>
    explicit data_set(matrix<real_type, layout> data_points);
    /**
     * @brief Create a new data set from the provided @p data_points and @p labels.
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the row-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the column-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    template <layout_type layout>
    data_set(matrix<real_type, layout> data_points, std::vector<label_type> labels);
    /**
     * @brief Create a new data set from the the provided @p data_points and scale them using the provided @p scale_parameter.
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] scale_parameter the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the row-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the column-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception all exceptions thrown by plssvm::data_set::scale
     */
    template <layout_type layout>
    data_set(matrix<real_type, layout> data_points, scaling scale_parameter);
    /**
     * @brief Create a new data set from the the provided @p data_points and @p labels and scale the @p data_points using the provided @p scale_parameter.
     * @note If the provided matrix isn't padded, adds the necessary padding entries automatically.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scale_parameter the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the row-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the column-padding is not equal to plssvm::PADDING_SIZE
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::data_set_exception all exceptions thrown by plssvm::data_set::scale
     */
    template <layout_type layout>
    data_set(matrix<real_type, layout> data_points, std::vector<label_type> labels, scaling scale_parameter);

    /**
     * @brief Save the data points and potential labels of this data set to the file @p filename using the file @p format type.
     * @param[in] filename the file to save the data points and labels to
     * @param[in] format the file format
     */
    void save(const std::string &filename, file_format_type format) const;
    /**
     * @brief Save the data points and potential labels of this data set to the file @p filename.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @param[in] filename the file to save the data points and labels to
     * @throws plssvm::data_set_exception if the file extension isn't one of `libsvm` or `arff`
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
     * @brief Returns an optional to the classes in this data set.
     * @details If the data set contains the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns the labels `{ -1, 1 }`.
     * @note Must not return a optional reference, since it would bind to a temporary!
     * @return if this data set contains labels, returns a reference to all classes, otherwise returns a `std::nullopt` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::optional<std::vector<label_type>> classes() const;

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
     * @brief Returns the number of classes in this data set.
     * @details If the data set contains the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns `2`.
     *          It is the same as: `dataset.classes()->size()`
     * @return the number of classes (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_classes() const noexcept { return mapping_ != nullptr ? mapping_->num_mappings() : 0; }

    /**
     * @brief Returns whether this data set has been scaled or not.
     * @details The used scaling factors can be retrieved using plssvm::data_set::scaling_factors().
     * @return `true` if this data set has been scaled, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_scaled() const noexcept { return scale_parameters_ != nullptr; }
    /**
     * @brief Returns the scaling factors as an optional reference used to scale the data points in this data set.
     * @details Can be used to scale another data set in the same way (e.g., a test data set).
     *          If the data set has been scaled, the scaling factors can be retrieved as using: `dataset.scaling_factors()->%get()`.
     * @return the scaling factors (`[[nodiscard]]`)
     */
    [[nodiscard]] optional_ref<const scaling> scaling_factors() const noexcept;

  private:
    /**
     * @brief Default construct an empty data set.
     */
    data_set() :
        data_ptr_{ std::make_shared<soa_matrix<real_type>>() } {}

    /**
     * @brief Create the mapping between the provided labels and the internally used indices.
     * @param[in] classes the list of different labels used to create the index mapping
     * @throws plssvm::data_set_exception any exception of the plssvm::data_set::label_mapper class
     */
    void create_mapping(const std::vector<label_type> &classes);
    /**
     * @brief Scale the feature values of the data set to the provided range.
     * @details Scales all data points feature wise, i.e., one scaling factor is responsible, e.g., for the first feature of **all** data points. <br>
     *          Scaling a data value \f$x\f$ to the range \f$[a, b]\f$ is done with the formular:
     *          \f$x_{scaled} = a + (b - a) \cdot \frac{x - min(x)}{max(x) - min(x)}\f$
     * @throws plssvm::data_set_exception if more scaling factors than features are present
     * @throws plssvm::data_set_exception if the largest scaling factor index is larger than the number of features
     * @throws plssvm::data_set_exception if for any feature more than one scaling factor is present
     */
    void scale();
    /**
     * @brief Read the data points and potential labels from the file @p filename assuming the plssvm::file_format_type @p format.
     * @param[in] filename the filename to read the data from
     * @param[in] format the assumed file format type
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by the respective functions in the plssvm::detail::io namespace
     * @throws plssvm::data_set_exception if labels are present in @p filename, all exceptions thrown by plssvm::data_set::create_mapping
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

    /// The mapping used to convert the original label to its mapped value and vice versa; may be `nullptr` if no labels have been provided.
    std::shared_ptr<const label_mapper> mapping_{ nullptr };
    /// The scaling parameters used to scale the data points in this data set; may be `nullptr` if no data point scaling was requested.
    std::shared_ptr<scaling> scale_parameters_{ nullptr };
};

//*************************************************************************************************************************************//
//                                                         scaling nested-class                                                        //
//*************************************************************************************************************************************//

/**
 * @brief Implements all necessary data and functions needed for scaling a plssvm::data_set to an user-defined range.
 */
template <typename U>
class data_set<U>::scaling {
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
         * @brief Construct new scaling factors struct with the provided values.
         * @param[in] feature_index the feature index for which the bounds are valid
         * @param[in] lower_bound the lowest value of the feature @p feature_index for all data points
         * @param[in] upper_bound the maximum value of the feature @p feature_index for all data points
         */
        factors(const size_type feature_index, const real_type lower_bound, const real_type upper_bound) :
            feature{ feature_index }, lower{ lower_bound }, upper{ upper_bound } {}

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
    scaling(real_type lower, real_type upper);
    /**
     * @brief Read the scaling interval and factors from the provided file @p filename.
     * @param[in] filename the filename to read the scaling information from
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by the plssvm::detail::io::parse_scaling_factors function
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

template <typename U>
data_set<U>::scaling::scaling(const real_type lower, const real_type upper) :
    scaling_interval{ std::make_pair(lower, upper) } {
    if (lower >= upper) {
        throw data_set_exception{ fmt::format("Inconsistent scaling interval specification: lower ({}) must be less than upper ({})!", lower, upper) };
    }
}

template <typename U>
data_set<U>::scaling::scaling(const std::string &filename) {
    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // read scaling values from file
    std::tie(scaling_interval, scaling_factors) = detail::io::parse_scaling_factors<factors>(reader);
}

template <typename U>
void data_set<U>::scaling::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // write scaling values to file
    detail::io::write_scaling_factors(filename, scaling_interval, scaling_factors);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Write {} scaling factors in {} to the file '{}'.\n",
                detail::tracking_entry{ "scaling_factors_write", "num_scaling_factors", scaling_factors.size() },
                detail::tracking_entry{ "scaling_factors_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking_entry{ "scaling_factors_write", "filename", filename });
}

//*************************************************************************************************************************************//
//                                                      label mapper nested-class                                                      //
//*************************************************************************************************************************************//

/**
 * @brief Implements all necessary functionality to map arbitrary labels to labels usable by the C-SVMs.
 * @details Currently maps all labels to { -1 , 1 }.
 */
template <typename U>
class data_set<U>::label_mapper {
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
data_set<U>::data_set::label_mapper::label_mapper(const std::vector<label_type> &classes) {
    PLSSVM_ASSERT(std::set(classes.cbegin(), classes.cend()).size() == classes.size(),
                  "The provided labels for the label_mapper must not include duplicated ones!");
    // create mapping
    std::size_t idx = 0;
    for (auto it = classes.begin(), end = classes.end(); it != end; ++it) {
        label_to_index_[*it] = idx;
        index_to_label_[idx] = *it;
        ++idx;
    }
}
/// @endcond

template <typename U>
auto data_set<U>::label_mapper::get_mapped_index_by_label(const label_type &label) const -> const size_type & {
    if (!detail::contains(label_to_index_, label)) {
        throw data_set_exception{ fmt::format("Label \"{}\" unknown in this label mapping!", label) };
    }
    return label_to_index_.at(label);
}

template <typename U>
auto data_set<U>::label_mapper::get_label_by_mapped_index(const size_type &mapped_index) const -> const label_type & {
    if (!detail::contains(index_to_label_, mapped_index)) {
        throw data_set_exception{ fmt::format("Mapped index \"{}\" unknown in this label mapping!", mapped_index) };
    }
    return index_to_label_.at(mapped_index);
}

template <typename U>
auto data_set<U>::label_mapper::num_mappings() const noexcept -> size_type {
    PLSSVM_ASSERT(label_to_index_.size() == index_to_label_.size(), "Both maps must contain the same number of values, but {} and {} were given!", label_to_index_.size(), index_to_label_.size());
    return label_to_index_.size();
}

template <typename U>
auto data_set<U>::label_mapper::labels() const -> std::vector<label_type> {
    std::vector<label_type> available_labels;
    available_labels.reserve(this->num_mappings());
    for (const auto &[key, value] : label_to_index_) {
        available_labels.push_back(key);
    }
    return available_labels;
}

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
data_set<U>::data_set(const std::string &filename, scaling scale_parameter) :
    data_set{ filename } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename U>
data_set<U>::data_set(const std::string &filename, file_format_type format, scaling scale_parameter) :
    data_set{ filename, format } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points) try :
    data_set{ soa_matrix<real_type>{ data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels) try :
    data_set{ soa_matrix<real_type>{ data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }, std::move(labels) } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points, scaling scale_parameter) try :
    data_set{ soa_matrix<real_type>{ data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }, std::move(scale_parameter) } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
data_set<U>::data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, scaling scale_parameter) try :
    data_set{ soa_matrix<real_type>{ data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }, std::move(labels), std::move(scale_parameter) } {}
    catch (const matrix_exception &e) {
        throw data_set_exception{ e.what() };
    }

template <typename U>
template <layout_type layout>
data_set<U>::data_set(matrix<real_type, layout> data_points) :
    num_data_points_{ data_points.num_rows() }, num_features_{ data_points.num_cols() }, data_ptr_{ std::make_shared<soa_matrix<real_type>>(data_points.is_padded() ? std::move(data_points) : matrix<real_type, layout>{ data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }) } {
    // the provided data points vector may not be empty
    if (data_ptr_->num_rows() == 0) {
        throw data_set_exception{ "Data vector is empty!" };
    }
    if (data_ptr_->num_cols() == 0) {
        throw data_set_exception{ "No features provided for the data points!" };
    }
    // the padding sizes must match the expected once! (otherwise the backend calculations would result in garbage)
    if (data_ptr_->padding()[0] != plssvm::PADDING_SIZE) {
        throw data_set_exception{ fmt::format("Expected {} as row-padding size, but got {}!", plssvm::PADDING_SIZE, data_ptr_->padding()[0]) };
    }
    if (data_ptr_->padding()[1] != plssvm::PADDING_SIZE) {
        throw data_set_exception{ fmt::format("Expected {} as column-padding size, but got {}!", plssvm::PADDING_SIZE, data_ptr_->padding()[1]) };
    }

    detail::log(verbosity_level::full | verbosity_level::timing,
                "Created a data set with {} data points and {} features.\n",
                detail::tracking_entry{ "data_set_create", "num_data_points", num_data_points_ },
                detail::tracking_entry{ "data_set_create", "num_features", num_features_ });
}

template <typename U>
template <layout_type layout>
data_set<U>::data_set(matrix<real_type, layout> data_points, std::vector<label_type> labels) :
    num_data_points_{ data_points.num_rows() }, num_features_{ data_points.num_cols() }, data_ptr_{ std::make_shared<soa_matrix<real_type>>(data_points.is_padded() ? std::move(data_points) : matrix<real_type, layout>{ data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }) }, labels_ptr_{ std::make_shared<std::vector<label_type>>(std::move(labels)) } {
    // the number of labels must be equal to the number of data points!
    if (data_ptr_->num_rows() != labels_ptr_->size()) {
        throw data_set_exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", labels_ptr_->size(), data_ptr_->num_rows()) };
    }
    // the padding sizes must match the expected once! (otherwise the backend calculations would result in garbage)
    if (data_ptr_->padding()[0] != plssvm::PADDING_SIZE) {
        throw data_set_exception{ fmt::format("Expected {} as row-padding size, but got {}!", plssvm::PADDING_SIZE, data_ptr_->padding()[0]) };
    }
    if (data_ptr_->padding()[1] != plssvm::PADDING_SIZE) {
        throw data_set_exception{ fmt::format("Expected {} as column-padding size, but got {}!", plssvm::PADDING_SIZE, data_ptr_->padding()[1]) };
    }

    // create mapping from labels
    std::set<label_type> unique_labels(labels_ptr_->cbegin(), labels_ptr_->cend());
    this->create_mapping(std::vector<label_type>(unique_labels.cbegin(), unique_labels.cend()));

    detail::log(verbosity_level::full | verbosity_level::timing,
                "Created a data set with {} data points, {} features, and {} classes.\n",
                detail::tracking_entry{ "data_set_create", "num_data_points", num_data_points_ },
                detail::tracking_entry{ "data_set_create", "num_features", num_features_ },
                detail::tracking_entry{ "data_set_create", "num_classes", this->num_classes() });
}

template <typename U>
template <layout_type layout>
data_set<U>::data_set(matrix<real_type, layout> data_points, scaling scale_parameter) :
    data_set{ std::move(data_points) } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename U>
template <layout_type layout>
data_set<U>::data_set(matrix<real_type, layout> data_points, std::vector<label_type> labels, scaling scale_parameter) :
    data_set{ std::move(data_points), std::move(labels) } {
    // initialize scaling
    scale_parameters_ = std::make_shared<scaling>(std::move(scale_parameter));
    // scale data set
    this->scale();
}

template <typename U>
void data_set<U>::save(const std::string &filename, const file_format_type format) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

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

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Write {} data points with {} features and {} classes in {} to the {} file '{}'.\n",
                detail::tracking_entry{ "data_set_write", "num_data_points", num_data_points_ },
                detail::tracking_entry{ "data_set_write", "num_features", num_features_ },
                detail::tracking_entry{ "data_set_write", "num_classes", this->num_classes() },
                detail::tracking_entry{ "data_set_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking_entry{ "data_set_write", "format", format },
                detail::tracking_entry{ "data_set_write", "filename", filename });
}

template <typename U>
void data_set<U>::save(const std::string &filename) const {
    if (detail::ends_with(filename, ".libsvm")) {
        this->save(filename, file_format_type::libsvm);
    } else if (detail::ends_with(filename, ".arff")) {
        this->save(filename, file_format_type::arff);
    } else {
        throw data_set_exception(fmt::format("Unrecognized file extension for file \"{}\" (must be one of: .libsvm or .arff)!", filename));
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
auto data_set<U>::classes() const -> std::optional<std::vector<label_type>> {
    if (this->has_labels()) {
        return std::make_optional(mapping_->labels());
    }
    return std::nullopt;
}

template <typename U>
auto data_set<U>::scaling_factors() const noexcept -> optional_ref<const scaling> {
    if (this->is_scaled()) {
        return std::make_optional(std::cref(*scale_parameters_));
    }
    return std::nullopt;
}

//*************************************************************************************************************************************//
//                                                      PRIVATE MEMBER FUNCTIONS                                                       //
//*************************************************************************************************************************************//

template <typename U>
void data_set<U>::create_mapping(const std::vector<label_type> &classes) {
    PLSSVM_ASSERT(labels_ptr_ != nullptr, "Can't create mapping if no labels are provided!");

    // create label mapping
    label_mapper mapper{ classes };

    // convert input labels to now mapped values
    aos_matrix<real_type> tmp{ mapper.num_mappings(), labels_ptr_->size(), real_type{ -1.0 } };

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

template <typename U>
void data_set<U>::scale() {
    PLSSVM_ASSERT(this->is_scaled(), "No scaling parameters given for scaling!");

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // unpack scaling interval pair
    const real_type lower = scale_parameters_->scaling_interval.first;
    const real_type upper = scale_parameters_->scaling_interval.second;

    // calculate scaling factors if necessary, use provided ones otherwise
    if (scale_parameters_->scaling_factors.empty()) {
        // calculate feature-wise min/max values for scaling
        for (size_type feature = 0; feature < num_features_; ++feature) {
            real_type min_value = std::numeric_limits<real_type>::max();
            real_type max_value = std::numeric_limits<real_type>::lowest();

            // calculate min/max values of all data points at the specific feature
            #pragma omp parallel for default(shared) firstprivate(feature) reduction(min : min_value) reduction(max : max_value)
            for (size_type data_point = 0; data_point < num_data_points_; ++data_point) {
                min_value = std::min(min_value, (*data_ptr_)(data_point, feature));
                max_value = std::max(max_value, (*data_ptr_)(data_point, feature));
            }

            // add scaling factor only if min_value != 0.0 AND max_value != 0.0
            if (!(min_value == real_type{ 0.0 } && max_value == real_type{ 0.0 })) {
                scale_parameters_->scaling_factors.emplace_back(feature, min_value, max_value);
            }
        }
    } else {
        // the number of scaling factors may not exceed the number of features
        if (scale_parameters_->scaling_factors.size() > num_features_) {
            throw data_set_exception{ fmt::format("Need at most as much scaling factors as features in the data set are present ({}), but {} were given!", num_features_, scale_parameters_->scaling_factors.size()) };
        }
        // sort vector
        const auto scaling_factors_comp_less = [](const typename scaling::factors &lhs, const typename scaling::factors &rhs) { return lhs.feature < rhs.feature; };
        std::sort(scale_parameters_->scaling_factors.begin(), scale_parameters_->scaling_factors.end(), scaling_factors_comp_less);
        // check whether the biggest feature index is smaller than the number of features
        if (scale_parameters_->scaling_factors.back().feature >= num_features_) {
            throw data_set_exception{ fmt::format("The maximum scaling feature index most not be greater than {}, but is {}!", num_features_ - 1, scale_parameters_->scaling_factors.back().feature) };
        }
        // check that there are no duplicate entries
        const auto scaling_factors_comp_eq = [](const typename scaling::factors &lhs, const typename scaling::factors &rhs) { return lhs.feature == rhs.feature; };
        const auto iter = std::adjacent_find(scale_parameters_->scaling_factors.begin(), scale_parameters_->scaling_factors.end(), scaling_factors_comp_eq);
        if (iter != scale_parameters_->scaling_factors.end()) {
            throw data_set_exception{ fmt::format("Found more than one scaling factor for the feature index {}!", iter->feature) };
        }
    }

    // scale values
    #pragma omp parallel for default(shared) firstprivate(lower, upper)
    for (size_type i = 0; i < scale_parameters_->scaling_factors.size(); ++i) {
        // extract feature-wise min and max values
        const typename scaling::factors factor = scale_parameters_->scaling_factors[i];
        // scale data values
        for (size_type data_point = 0; data_point < num_data_points_; ++data_point) {
            (*data_ptr_)(data_point, factor.feature) = lower + (upper - lower) * ((*data_ptr_)(data_point, factor.feature) - factor.lower) / (factor.upper - factor.lower);
        }
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Scaled the data set to the range [{}, {}] in {}.\n",
                detail::tracking_entry{ "data_set_scale", "lower", lower },
                detail::tracking_entry{ "data_set_scale", "upper", upper },
                detail::tracking_entry{ "data_set_scale", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });
}

template <typename U>
void data_set<U>::read_file(const std::string &filename, file_format_type format) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

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

    // create label mapping
    if (this->has_labels()) {
        std::set<label_type> unique_labels(labels_ptr_->cbegin(), labels_ptr_->cend());
        this->create_mapping(std::vector<label_type>(unique_labels.cbegin(), unique_labels.cend()));
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Read {} data points with {} features and {} classes in {} using the {} parser from file '{}'.\n",
                detail::tracking_entry{ "data_set_read", "num_data_points", num_data_points_ },
                detail::tracking_entry{ "data_set_read", "num_features", num_features_ },
                detail::tracking_entry{ "data_set_read", "num_classes", this->num_classes() },
                detail::tracking_entry{ "data_set_read", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking_entry{ "data_set_read", "format", format },
                detail::tracking_entry{ "data_set_read", "filename", filename });
}

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_HPP_
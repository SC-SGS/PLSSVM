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

#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/data_set/data_set.hpp"                    // plssvm::data_set
#include "plssvm/data_set/min_max_scaler.hpp"              // plssvm::min_max_scaler
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/type_list.hpp"                     // plssvm::detail::{supported_label_types_regression, tuple_contains_v}
#include "plssvm/file_format_types.hpp"                    // plssvm::file_format_type
#include "plssvm/matrix.hpp"                               // plssvm::aos_matrix
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/shape.hpp"                                // plssvm::shape
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include <chrono>   // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>  // std::size_t
#include <memory>   // std::make_shared
#include <string>   // std::string
#include <utility>  // std::forward
#include <utility>  // std::move
#include <vector>   // std::vector

namespace plssvm {

// forward declare C-SVR class
class csvr;

/**
 * @example regression_data_set_examples.cpp
 * @brief A few examples regarding the plssvm::regression_data_set_examples class.
 */

/**
 * @brief Encapsulate all necessary data that is needed for training or predicting using an C-SVR.
 * @details May or may not contain labels!
 *          Internally, saves all data using [`std::shared_ptr`](https://en.cppreference.com/w/cpp/memory/shared_ptr) to make a plssvm::regression_data_set relatively cheap to copy!
 * @tparam U the label type of the data (must be an arithmetic type, except boolean or character types; default: `real_type`)
 */
template <typename U = real_type>
class regression_data_set : public data_set<U> {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<U, detail::supported_label_types_regression>,
                  "Illegal label type for regression provided! See the 'supported_label_types_regression' in the type_list.hpp header for a list of the allowed types.");

    // befriend C-SVR class used with the regression data set: necessary to access `data_ptr_`, `mapping_`, and `y_ptr_`
    friend class csvr;

    /// The base data set class.
    using base_data_set = data_set<U>;

    using base_data_set::creation_start_time_;
    using base_data_set::labels_ptr_;
    using base_data_set::y_ptr_;

  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using typename base_data_set::label_type;
    /// An unsigned integer type.
    using typename base_data_set::size_type;
    // Make the overloaded non-virtual save member function visible.
    using base_data_set::save;
    /// The C-SVM type used with this data set.
    using svm_fit_type = ::plssvm::csvr;

    /**
     * @brief Read the data points from the file @p filename.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] filename the file to read the data points from
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     */
    explicit regression_data_set(const std::string &filename) :
        base_data_set{ mpi::communicator{}, filename } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the file to read the data points from
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     */
    regression_data_set(mpi::communicator comm, const std::string &filename) :
        base_data_set{ std::move(comm), filename } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the @p plssvm::file_format_type.
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     */
    regression_data_set(const std::string &filename, file_format_type format) :
        base_data_set{ mpi::communicator{}, filename, format } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the @p plssvm::file_format_type.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     */
    regression_data_set(mpi::communicator comm, const std::string &filename, file_format_type format) :
        base_data_set{ std::move(comm), filename, format } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename and scale it using the provided @p scaler.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] filename the file to read the data points from
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    regression_data_set(const std::string &filename, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, filename, std::move(scaler) } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename and scale it using the provided @p scaler.
     *        Automatically determines the plssvm::file_format_type based on the file extension.
     * @details If @p filename ends with `.arff` it uses the ARFF parser, otherwise the LIBSVM parser is used.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the file to read the data points from
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    regression_data_set(mpi::communicator comm, const std::string &filename, min_max_scaler scaler) :
        base_data_set{ std::move(comm), filename, std::move(scaler) } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the plssvm::file_format_type @p format and
     *        scale it using the provided @p scaler.
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    regression_data_set(const std::string &filename, file_format_type format, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, filename, format, std::move(scaler) } { this->init(); }

    /**
     * @brief Read the data points from the file @p filename assuming that the file is given in the plssvm::file_format_type @p format and
     *        scale it using the provided @p scaler.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the file to read the data points from
     * @param[in] format the assumed file format used to parse the data points
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::data_set::read_file
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    regression_data_set(mpi::communicator comm, const std::string &filename, file_format_type format, min_max_scaler scaler) :
        base_data_set{ std::move(comm), filename, format, std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    explicit regression_data_set(const std::vector<std::vector<real_type>> &data_points) :
        base_data_set{ mpi::communicator{}, data_points } { this->init(); }

    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points) :
        base_data_set{ std::move(comm), data_points } { this->init(); }

    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix and copying the @p labels.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    regression_data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels) } { this->init(); }

    /**
     * @brief Create a new data set by converting the provided @p data_points to a plssvm::matrix and copying the @p labels.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels) :
        base_data_set{ std::move(comm), data_points, std::move(labels) } { this->init(); }

    /**
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and scale them using the provided @p scaler.
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    regression_data_set(const std::vector<std::vector<real_type>> &data_points, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and scale them using the provided @p scaler.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(scaler) } { this->init(); }

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
    regression_data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set  by converting the provided @p data_points to a plssvm::matrix and copying the @p labels and scale the @p data_points using the provided @p scaler.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set from the provided @p data_points.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    template <layout_type layout>
    explicit regression_data_set(const matrix<real_type, layout> &data_points) :
        base_data_set{ mpi::communicator{}, data_points } { this->init(); }

    /**
     * @brief Create a new data set from the provided @p data_points.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @tparam layout the layout type of the input matrix
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points) :
        base_data_set{ std::move(comm), data_points } { this->init(); }

    /**
     * @brief Create a new data set from the provided @p data_points and @p labels.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    template <layout_type layout>
    regression_data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels) } { this->init(); }

    /**
     * @brief Create a new data set from the provided @p data_points and @p labels.
     * @tparam layout the layout type of the input matrix
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points, std::vector<label_type> labels) :
        base_data_set{ std::move(comm), data_points, std::move(labels) } { this->init(); }

    /**
     * @brief Create a new data set from the the provided @p data_points and scale them using the provided @p scaler.
     * @tparam layout the layout type of the input matrix
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    template <layout_type layout>
    regression_data_set(const matrix<real_type, layout> &data_points, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set from the the provided @p data_points and scale them using the provided @p scaler.
     * @tparam layout the layout type of the input matrix
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set from the the provided @p data_points and @p labels and scale the @p data_points using the provided @p scaler.
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
    regression_data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @brief Create a new data set from the the provided @p data_points and @p labels and scale the @p data_points using the provided @p scaler.
     * @tparam layout the layout type of the input matrix
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @brief Use the provided @p data_points in this data set.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    explicit regression_data_set(aos_matrix<real_type> &&data_points) :
        base_data_set{ mpi::communicator{}, std::move(data_points) } { this->init(); }

    /**
     * @brief Use the provided @p data_points in this data set.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     */
    regression_data_set(mpi::communicator comm, aos_matrix<real_type> &&data_points) :
        base_data_set{ std::move(comm), std::move(data_points) } { this->init(); }

    /**
     * @brief Use the provided @p data_points and @p labels in this data set.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    regression_data_set(aos_matrix<real_type> &&data_points, std::vector<label_type> &&labels) :
        base_data_set{ mpi::communicator{}, std::move(data_points), std::move(labels) } { this->init(); }

    /**
     * @brief Use the provided @p data_points and @p labels in this data set.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     */
    regression_data_set(mpi::communicator comm, aos_matrix<real_type> &&data_points, std::vector<label_type> &&labels) :
        base_data_set{ std::move(comm), std::move(data_points), std::move(labels) } { this->init(); }

    /**
     * @brief Use the provided @p data_points in this data set and scale them using the provided @p scaler.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    regression_data_set(aos_matrix<real_type> &&data_points, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, std::move(data_points), std::move(scaler) } { this->init(); }

    /**
     * @brief Use the provided @p data_points in this data set and scale them using the provided @p scaler.
     * @details Since no labels are provided, this data set may **not** be used to a call to plssvm::csvc::fit/plssvm::csvr::fit!
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    regression_data_set(mpi::communicator comm, aos_matrix<real_type> &&data_points, min_max_scaler scaler) :
        base_data_set{ std::move(comm), std::move(data_points), std::move(scaler) } { this->init(); }

    /**
     * @brief Use the provided @p data_points and @p labels in this data set and scale them using the provided @p scaler.
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     */
    regression_data_set(aos_matrix<real_type> &&data_points, std::vector<label_type> &&labels, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, std::move(data_points), std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @brief Use the provided @p data_points and @p labels in this data set and scale them using the provided @p scaler.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] data_points the data points used in this data set
     * @param[in] labels the labels used in this data set
     * @param[in] scaler the parameters used to scale the data set feature values to a given range
     * @throws plssvm::data_set_exception if the @p data_points vector is empty
     * @throws plssvm::data_set_exception if the data points in @p data_points have mismatching number of features
     * @throws plssvm::data_set_exception if any @p data_point has no features
     * @throws plssvm::data_set_exception if the number of data points in @p data_points and number of @p labels mismatch
     * @throws plssvm::min_max_scaler_exception all exceptions thrown by plssvm::min_max_scaler::scale
     * @throws plssvm::mpi_exception if the MPI communicator @p comm and the MPI communicator in @p scaler are not identical
     */
    regression_data_set(mpi::communicator comm, aos_matrix<real_type> &&data_points, std::vector<label_type> &&labels, min_max_scaler scaler) :
        base_data_set{ std::move(comm), std::move(data_points), std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::save
     */
    void save(const std::string &filename, file_format_type format) const override;

  private:
    /**
     * @brief Initialize the classification data set.
     */
    void init();

    /**
     * @copydoc plssvm::data_set::map_label
     */
    void map_label() override;
};

//*************************************************************************************************************************************//
//                                                      regression data set class                                                      //
//*************************************************************************************************************************************//

template <typename U>
void regression_data_set<U>::init() {
    // create label mapping
    if (this->has_labels()) {
        this->map_label();
    }

    const auto creation_end_time = std::chrono::steady_clock::now();
    const auto creation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(creation_end_time - creation_start_time_);

    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Created a regression data set with {} data points and {} features in {}.\n",
                detail::tracking::tracking_entry{ "data_set_create", "num_data_points", this->num_data_points() },
                detail::tracking::tracking_entry{ "data_set_create", "num_features", this->num_features() },
                detail::tracking::tracking_entry{ "data_set_create", "time", creation_duration });
}

template <typename U>
void regression_data_set<U>::save(const std::string &filename, const file_format_type format) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // save the data set
    base_data_set::save(filename, format);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Write {} regression data points with {} features in {} to the {} file '{}'.\n",
                detail::tracking::tracking_entry{ "data_set_write", "num_data_points", this->num_data_points() },
                detail::tracking::tracking_entry{ "data_set_write", "num_features", this->num_features() },
                detail::tracking::tracking_entry{ "data_set_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "data_set_write", "format", format },
                detail::tracking::tracking_entry{ "data_set_write", "filename", filename });
}

//*************************************************************************************************************************************//
//                                                      PRIVATE MEMBER FUNCTIONS                                                       //
//*************************************************************************************************************************************//

template <typename U>
void regression_data_set<U>::map_label() {
    PLSSVM_ASSERT(labels_ptr_ != nullptr, "Can't create mapping if no labels are provided!");

    // convert input labels to now mapped values
    std::vector<real_type> labels(labels_ptr_->size());
#pragma omp parallel for
    for (std::size_t i = 0; i < labels.size(); ++i) {
        labels[i] = static_cast<real_type>((*labels_ptr_)[i]);
    }

    y_ptr_ = std::make_shared<aos_matrix<real_type>>(shape{ 1, labels.size() }, labels);
}

}  // namespace plssvm

#endif  // PLSSVM_DATA_SET_REGRESSION_DATA_SET_HPP_

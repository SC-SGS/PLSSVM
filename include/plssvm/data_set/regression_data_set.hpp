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
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
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
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &)
     */
    explicit regression_data_set(const std::string &filename) :
        base_data_set{ mpi::communicator{}, filename } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &)
     */
    regression_data_set(mpi::communicator comm, const std::string &filename) :
        base_data_set{ std::move(comm), filename } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &, file_format_type)
     */
    regression_data_set(const std::string &filename, file_format_type format) :
        base_data_set{ mpi::communicator{}, filename, format } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &, file_format_type)
     */
    regression_data_set(mpi::communicator comm, const std::string &filename, file_format_type format) :
        base_data_set{ std::move(comm), filename, format } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &, min_max_scaler)
     */
    regression_data_set(const std::string &filename, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, filename, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &, min_max_scaler)
     */
    regression_data_set(mpi::communicator comm, const std::string &filename, min_max_scaler scaler) :
        base_data_set{ std::move(comm), filename, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &, file_format_type, min_max_scaler)
     */
    regression_data_set(const std::string &filename, file_format_type format, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, filename, format, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::string &, file_format_type, min_max_scaler)
     */
    regression_data_set(mpi::communicator comm, const std::string &filename, file_format_type format, min_max_scaler scaler) :
        base_data_set{ std::move(comm), filename, format, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &)
     */
    explicit regression_data_set(const std::vector<std::vector<real_type>> &data_points) :
        base_data_set{ mpi::communicator{}, data_points } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &)
     */
    explicit regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points) :
        base_data_set{ std::move(comm), data_points } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &, std::vector<label_type>)
     */
    regression_data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &, std::vector<label_type>)
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels) :
        base_data_set{ std::move(comm), data_points, std::move(labels) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &, min_max_scaler)
     */
    regression_data_set(const std::vector<std::vector<real_type>> &data_points, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &, min_max_scaler)
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &, std::vector<label_type>, min_max_scaler)
     */
    regression_data_set(const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const std::vector<std::vector<real_type>> &, std::vector<label_type>, min_max_scaler)
     */
    regression_data_set(mpi::communicator comm, const std::vector<std::vector<real_type>> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &)
     */
    template <layout_type layout>
    explicit regression_data_set(const matrix<real_type, layout> &data_points) :
        base_data_set{ mpi::communicator{}, data_points } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &)
     */
    template <layout_type layout>
    explicit regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points) :
        base_data_set{ std::move(comm), data_points } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &, std::vector<label_type>)
     */
    template <layout_type layout>
    regression_data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &, std::vector<label_type>)
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points, std::vector<label_type> labels) :
        base_data_set{ std::move(comm), data_points, std::move(labels) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &, min_max_scaler)
     */
    template <layout_type layout>
    regression_data_set(const matrix<real_type, layout> &data_points, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &, min_max_scaler)
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &, std::vector<label_type>, min_max_scaler)
     */
    template <layout_type layout>
    regression_data_set(const matrix<real_type, layout> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, const matrix<real_type, layout> &, std::vector<label_type>, min_max_scaler)
     */
    template <layout_type layout>
    regression_data_set(mpi::communicator comm, const matrix<real_type, layout> &data_points, std::vector<label_type> labels, min_max_scaler scaler) :
        base_data_set{ std::move(comm), data_points, std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&)
     */
    explicit regression_data_set(soa_matrix<real_type> &&data_points) :
        base_data_set{ mpi::communicator{}, std::move(data_points) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&)
     */
    explicit regression_data_set(mpi::communicator comm, soa_matrix<real_type> &&data_points) :
        base_data_set{ std::move(comm), std::move(data_points) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&, std::vector<label_type> &&)
     */
    regression_data_set(soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels) :
        base_data_set{ mpi::communicator{}, std::move(data_points), std::move(labels) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&, std::vector<label_type> &&)
     */
    regression_data_set(mpi::communicator comm, soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels) :
        base_data_set{ std::move(comm), std::move(data_points), std::move(labels) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&, min_max_scaler)
     */
    regression_data_set(soa_matrix<real_type> &&data_points, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, std::move(data_points), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&, min_max_scaler)
     */
    regression_data_set(mpi::communicator comm, soa_matrix<real_type> &&data_points, min_max_scaler scaler) :
        base_data_set{ std::move(comm), std::move(data_points), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&, std::vector<label_type> &&, min_max_scaler)
     */
    regression_data_set(soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels, min_max_scaler scaler) :
        base_data_set{ mpi::communicator{}, std::move(data_points), std::move(labels), std::move(scaler) } { this->init(); }

    /**
     * @copydoc plssvm::data_set::data_set(mpi::communicator, soa_matrix<real_type> &&, std::vector<label_type> &&, min_max_scaler)
     */
    regression_data_set(mpi::communicator comm, soa_matrix<real_type> &&data_points, std::vector<label_type> &&labels, min_max_scaler scaler) :
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

    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Created a regression data set with {} data points and {} features.\n",
                detail::tracking::tracking_entry{ "data_set_create", "num_data_points", this->num_data_points() },
                detail::tracking::tracking_entry{ "data_set_create", "num_features", this->num_features() });
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

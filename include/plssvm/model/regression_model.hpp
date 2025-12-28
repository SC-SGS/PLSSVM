/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a model class encapsulating the results of a C-SVR fit call.
 */

#ifndef PLSSVM_MODEL_REGRESSION_MODEL_HPP_
#define PLSSVM_MODEL_REGRESSION_MODEL_HPP_
#pragma once

#include "plssvm/constants.hpp"                                  // plssvm::real_type
#include "plssvm/data_set/regression_data_set.hpp"               // plssvm::regression_data_set
#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // plssvm::detail::io::{parse_libsvm_model_header_regression, parse_libsvm_model_data_regression, write_libsvm_model_data_regression}
#include "plssvm/detail/logging/mpi_log.hpp"                     // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"        // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/type_list.hpp"                           // plssvm::detail::{supported_label_types, tuple_contains_v}
#include "plssvm/matrix.hpp"                                     // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/model/model.hpp"                                // plssvm::model
#include "plssvm/mpi/communicator.hpp"                           // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                                  // plssvm::parameter
#include "plssvm/verbosity_levels.hpp"                           // plssvm::verbosity_level

#include <chrono>   // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr, std::make_shared
#include <string>   // std::string
#include <tuple>    // std::tie
#include <utility>  // std::move

// forward declare svr dummy struct used in the plssvm.SVR Python bindings
struct svr;

namespace plssvm {

/**
 * @example regression_model_examples.cpp
 * @brief A few examples regarding the plssvm::regression_model class.
 */

/**
 * @brief Implements a class encapsulating the result of a call to the C-SVR fit function. A model is used to predict the labels of a new data set.
 * @tparam U the type of the used labels (must be an arithmetic type, except boolean or character types; default: `real_type`)
 */
template <typename U = real_type>
class regression_model : public model<U> {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<U, detail::supported_label_types_regression>,
                  "Illegal label type for regression provided! See the 'supported_label_types_regression' in the type_list.hpp header for a list of the allowed types.");

    // befriend C-SVR class used with the regression data set: necessary to access the private constructor and multiple member variables
    friend class csvr;

    // befriend svr dummy struct used in the plssvm.SVR Python bindings
    friend struct ::svr;

    /// The base model class.
    using base_model = model<U>;

    // Make the protected member variables visible in the derived class.
    using base_model::alpha_ptr_;
    using base_model::data_;
    using base_model::num_features_;
    using base_model::num_support_vectors_;
    using base_model::params_;
    using base_model::rho_ptr_;

  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using label_type = U;
    /// The unsigned size type.
    using size_type = std::size_t;

    /**
     * @brief Read a previously learned model from the LIBSVM model file @p filename.
     * @param[in] filename the model file to read
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::detail::io::parse_libsvm_model_header and plssvm::detail::io::parse_libsvm_data
     */
    explicit regression_model(const std::string &filename);

    /**
     * @brief Read a previously learned model from the LIBSVM model file @p filename.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the model file to read
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::detail::io::parse_libsvm_model_header and plssvm::detail::io::parse_libsvm_data
     */
    regression_model(mpi::communicator comm, const std::string &filename);

    /**
     * @brief Save the model to a LIBSVM model file for later usage.
     * @param[in] filename the file to save the model to
     */
    void save(const std::string &filename) const override;

  private:
    /**
     * @brief Create a new model using the SVM parameter @p params and the @p data.
     * @details Default initializes the weights, i.e., no weights have currently been learned.
     * @note This constructor may only be used in the befriended base C-SVM class!
     * @param[in] params the SVM parameters used to learn this model
     * @param[in] data the data used to learn this model
     */
    regression_model(parameter params, regression_data_set<label_type> data);
};

template <typename U>
regression_model<U>::regression_model(parameter params, regression_data_set<label_type> data) :
    base_model{ std::move(params), std::make_shared<regression_data_set<label_type>>(std::move(data)) } { }

template <typename U>
regression_model<U>::regression_model(const std::string &filename) :
    regression_model{ mpi::communicator{}, filename } { }

template <typename U>
regression_model<U>::regression_model(mpi::communicator comm, const std::string &filename) :
    base_model{ std::move(comm) } {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // parse the libsvm model header
    std::size_t num_header_lines{};
    std::tie(params_, *rho_ptr_, num_header_lines) = detail::io::parse_libsvm_model_header_regression(reader.lines());

    // create empty support vectors and alpha vector
    soa_matrix<real_type> support_vectors{};

    // parse libsvm model data
    std::tie(support_vectors, *alpha_ptr_) = detail::io::parse_libsvm_model_data_regression(reader, num_header_lines);
    num_support_vectors_ = support_vectors.num_rows();
    num_features_ = support_vectors.num_cols();

    // create data set
    const verbosity_level old_verbosity = verbosity;
    verbosity = verbosity_level::quiet;
    data_ = std::make_shared<regression_data_set<label_type>>(std::move(support_vectors));
    verbosity = old_verbosity;

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Read {} support vectors with {} features in {} using the libsvm regression model parser from file '{}'.\n\n",
                detail::tracking::tracking_entry{ "model_read", "num_support_vectors", this->num_support_vectors() },
                detail::tracking::tracking_entry{ "model_read", "num_features", this->num_features() },
                detail::tracking::tracking_entry{ "model_read", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "model_read", "filename", filename });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "model_read", "rho", this->rho() }));
}

template <typename U>
void regression_model<U>::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    if (this->communicator().is_main_rank()) {
        // save model file header and support vectors
        detail::io::write_libsvm_model_data_regression(filename, this->communicator(), this->get_params(), this->rho(), this->weights(), dynamic_cast<regression_data_set<label_type> &>(*data_));
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Write {} support vectors with {} features in {} to the libsvm regression model file '{}'.\n",
                detail::tracking::tracking_entry{ "model_write", "num_support_vectors", this->num_support_vectors() },
                detail::tracking::tracking_entry{ "model_write", "num_features", this->num_features() },
                detail::tracking::tracking_entry{ "model_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "model_write", "filename", filename });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "model_write", "rho", this->rho() }));
}

}  // namespace plssvm

#endif  // PLSSVM_MODEL_REGRESSION_MODEL_HPP_

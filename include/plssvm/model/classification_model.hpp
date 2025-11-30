/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a model class encapsulating the results of a C-SVC fit call.
 */

#ifndef PLSSVM_MODEL_CLASSIFICATION_MODEL_HPP_
#define PLSSVM_MODEL_CLASSIFICATION_MODEL_HPP_
#pragma once

#include "plssvm/classification_types.hpp"                           // plssvm::classification_type
#include "plssvm/constants.hpp"                                      // plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"               // plssvm::classification_data_set
#include "plssvm/detail/assert.hpp"                                  // PLSSVM_ASSERT
#include "plssvm/detail/io/classification_libsvm_model_parsing.hpp"  // plssvm::detail::io::{parse_libsvm_model_header_classification, parse_libsvm_model_data_classification, write_libsvm_model_data_classification}
#include "plssvm/detail/io/file_reader.hpp"                          // plssvm::detail::io::file_reader
#include "plssvm/detail/logging/mpi_log.hpp"                         // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"            // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/type_list.hpp"                               // plssvm::detail::{supported_label_types, tuple_contains_v}
#include "plssvm/exceptions/exceptions.hpp"                          // plssvm::exception
#include "plssvm/matrix.hpp"                                         // plssvm::soa_matrix
#include "plssvm/model/model.hpp"                                    // plssvm::model
#include "plssvm/mpi/communicator.hpp"                               // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                                      // plssvm::parameter
#include "plssvm/verbosity_levels.hpp"                               // plssvm::verbosity_level

#include <chrono>   // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr, std::make_shared
#include <numeric>  // std::iota
#include <string>   // std::string
#include <tuple>    // std::tie
#include <utility>  // std::move
#include <vector>   // std::vector

// forward declare svc dummy struct used in the plssvm.SVC Python bindings
struct svc;

namespace plssvm {

/**
 * @example classification_model_examples.cpp
 * @brief A few examples regarding the plssvm::classification_model class.
 */

/**
 * @brief Implements a class encapsulating the result of a call to the C-SVC fit function. A model is used to predict the labels of a new data set.
 * @tparam U the type of the used labels (must be an arithmetic type or `std:string`; default: `int`)
 */
template <typename U = int>
class classification_model : public model<U> {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<U, detail::supported_label_types_classification>,
                  "Illegal label type for classification provided! See the 'supported_label_types_classification' in the type_list.hpp header for a list of the allowed types.");

    // befriend C-SVC class used with the classification data set: necessary to access the private constructor and multiple member variables
    friend class csvc;

    // befriend svc dummy struct used in the plssvm.SVC Python bindings
    friend struct ::svc;

    /// The base model class.
    using base_model = model<U>;

    // Make the protected member variables visible in the derived class.
    using base_model::alpha_ptr_;
    using base_model::data_;
    using base_model::num_features_;
    using base_model::num_support_vectors_;
    using base_model::params_;
    using base_model::rho_ptr_;
    using base_model::w_ptr_;

  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using typename base_model::label_type;
    /// An unsigned integer type.
    using typename base_model::size_type;

    /**
     * @brief Read a previously learned model from the LIBSVM model file @p filename.
     * @param[in] filename the model file to read
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::detail::io::parse_libsvm_model_header and plssvm::detail::io::parse_libsvm_data
     */
    explicit classification_model(const std::string &filename);

    /**
     * @brief Read a previously learned model from the LIBSVM model file @p filename.
     * @param[in] comm the used MPI communicator (**note**: current only used to restrict logging outputs to the main MPI rank)
     * @param[in] filename the model file to read
     * @throws plssvm::invalid_file_format_exception all exceptions thrown by plssvm::detail::io::parse_libsvm_model_header and plssvm::detail::io::parse_libsvm_data
     */
    classification_model(mpi::communicator comm, const std::string &filename);

    /**
     * @brief Save the model to a LIBSVM model file for later usage.
     * @param[in] filename the file to save the model to
     */
    void save(const std::string &filename) const override;

    /**
     * @brief Returns the number of classes in this model.
     * @details If the data set contains the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns `2`.
     *          It is the same as: `model.classes().size()`
     * @return the number of classes (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_classes() const noexcept { return dynamic_cast<const classification_data_set<label_type> &>(*data_).num_classes(); }

    /**
     * @brief Returns the classes of the support vectors.
     * @details If the support vectors contain the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns the classes `{ -1, 1 }`.
     * @return all classes (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<label_type> classes() const {
        const auto classes_opt = dynamic_cast<const classification_data_set<label_type> &>(*data_).classes();
        if (!classes_opt.has_value()) {
            throw exception{ "No classes provided (this should NEVER be the case)!" };
        }
        return classes_opt.value();
    }

    /**
     * @brief Returns the multi-class classification strategy used to ft this model.
     * @return the multi-class classification strategy (`[[nodiscard]]`)
     */
    [[nodiscard]] classification_type get_classification_type() const noexcept { return classification_strategy_; }

  private:
    /**
     * @brief Create a new model using the SVM parameter @p params and the @p data.
     * @details Default initializes the weights, i.e., no weights have currently been learned.
     * @note This constructor may only be used in the befriended base C-SVM class!
     * @param[in] params the SVM parameters used to learn this model
     * @param[in] data the data used to learn this model
     * @param[in] classification_strategy the classification strategy used to fit this model
     */
    classification_model(parameter params, classification_data_set<label_type> data, classification_type classification_strategy);

    /// The classification strategy (one vs. all or one vs. one) used to fit this model.
    classification_type classification_strategy_{};

    /**
     * @brief For each class, holds the indices of all data points in the support vectors.
     * @details Unused for one vs. all classification.
     * @note Must be initialized to an empty vector instead of a `nullptr`.
     */
    std::shared_ptr<std::vector<std::vector<std::size_t>>> index_sets_ptr_{ std::make_shared<std::vector<std::vector<std::size_t>>>() };
};

template <typename U>
classification_model<U>::classification_model(parameter params, classification_data_set<label_type> data, const classification_type classification_strategy) :
    base_model{ std::move(params), std::make_shared<classification_data_set<label_type>>(std::move(data)) },
    classification_strategy_{ classification_strategy } { }

template <typename U>
classification_model<U>::classification_model(const std::string &filename) :
    classification_model{ mpi::communicator{}, filename } { }

template <typename U>
classification_model<U>::classification_model(mpi::communicator comm, const std::string &filename) :
    base_model{ std::move(comm) } {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // parse the libsvm model header
    std::vector<label_type> labels{};
    std::vector<label_type> unique_labels{};
    std::vector<std::size_t> num_sv_per_class{};
    std::size_t num_header_lines{};
    std::tie(params_, *rho_ptr_, labels, unique_labels, num_sv_per_class, num_header_lines) = detail::io::parse_libsvm_model_header_classification<label_type>(reader.lines());

    // create empty support vectors and alpha vector
    soa_matrix<real_type> support_vectors{};

    // parse libsvm model data
    std::tie(support_vectors, *alpha_ptr_, classification_strategy_) = detail::io::parse_libsvm_model_data_classification(reader, num_sv_per_class, num_header_lines);
    num_support_vectors_ = support_vectors.num_rows();
    num_features_ = support_vectors.num_cols();

    switch (classification_strategy_) {
        case classification_type::oaa:
            // empty index set for the OAA classification
            index_sets_ptr_ = std::make_shared<std::vector<std::vector<std::size_t>>>();
            break;
        case classification_type::oao:
            {
                // fill index_sets -> support vectors are sorted!
                index_sets_ptr_ = std::make_shared<std::vector<std::vector<std::size_t>>>(unique_labels.size());
                std::size_t running_idx{ 0 };
                for (std::size_t i = 0; i < num_sv_per_class.size(); ++i) {
                    (*index_sets_ptr_)[i] = std::vector<std::size_t>(num_sv_per_class[i]);
                    std::iota((*index_sets_ptr_)[i].begin(), (*index_sets_ptr_)[i].end(), running_idx);
                    running_idx += num_sv_per_class[i];
                }
            }
            break;
    }

    // create data set
    PLSSVM_ASSERT(support_vectors.num_rows() == labels.size(), "Number of labels ({}) must match the number of data points ({})!", labels.size(), support_vectors.num_rows());
    const verbosity_level old_verbosity = verbosity;
    verbosity = verbosity_level::quiet;
    data_ = std::make_shared<classification_data_set<label_type>>(std::move(support_vectors), std::move(labels));
    verbosity = old_verbosity;

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Read {} support vectors with {} features and {} classes using {} classification in {} using the libsvm classification model parser from file '{}'.\n\n",
                detail::tracking::tracking_entry{ "model_read", "num_support_vectors", this->num_support_vectors() },
                detail::tracking::tracking_entry{ "model_read", "num_features", this->num_features() },
                detail::tracking::tracking_entry{ "model_read", "num_classes", this->num_classes() },
                classification_type_to_full_string(classification_strategy_),
                detail::tracking::tracking_entry{ "model_read", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "model_read", "filename", filename });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "model_read", "rho", this->rho() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "model_read", "classification_type", this->get_classification_type() }));
}

template <typename U>
void classification_model<U>::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    if (this->communicator().is_main_rank()) {
        // save model file header and support vectors
        detail::io::write_libsvm_model_data_classification(filename, this->communicator(), this->get_params(), this->get_classification_type(), this->rho(), this->weights(), *index_sets_ptr_, dynamic_cast<classification_data_set<label_type> &>(*data_));
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                this->communicator(),
                "Write {} support vectors with {} features and {} classes using {} classification in {} to the libsvm classification model file '{}'.\n",
                detail::tracking::tracking_entry{ "model_write", "num_support_vectors", this->num_support_vectors() },
                detail::tracking::tracking_entry{ "model_write", "num_features", this->num_features() },
                detail::tracking::tracking_entry{ "model_write", "num_classes", this->num_classes() },
                classification_type_to_full_string(this->get_classification_type()),
                detail::tracking::tracking_entry{ "model_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking::tracking_entry{ "model_write", "filename", filename });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "model_write", "rho", this->rho() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "model_write", "classification_type", this->get_classification_type() }));
}

}  // namespace plssvm

#endif  // PLSSVM_MODEL_CLASSIFICATION_MODEL_HPP_

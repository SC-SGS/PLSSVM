/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a model class encapsulating the results of a SVM fit call.
 */

#ifndef PLSSVM_MODEL_HPP_
#define PLSSVM_MODEL_HPP_
#pragma once

#include "plssvm/classification_types.hpp"            // plssvm::classification_type
#include "plssvm/constants.hpp"                       // plssvm::real_type
#include "plssvm/data_set.hpp"                        // plssvm::data_set
#include "plssvm/detail/assert.hpp"                   // PLSSVM_ASSERT
#include "plssvm/detail/io/libsvm_model_parsing.hpp"  // plssvm::detail::io::{parse_libsvm_model_header, parse_libsvm_model_data, write_libsvm_model_data}
#include "plssvm/detail/logging.hpp"                  // plssvm::detail::log
#include "plssvm/detail/performance_tracker.hpp"      // plssvm::detail::tracking_entry
#include "plssvm/detail/type_list.hpp"                // plssvm::detail::{supported_label_types, tuple_contains_v}
#include "plssvm/matrix.hpp"                          // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                       // plssvm::parameter
#include "plssvm/verbosity_levels.hpp"                // plssvm::verbosity_level

#include "fmt/chrono.h"  // format std::chrono types using fmt
#include "fmt/core.h"    // fmt::format

#include <chrono>    // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>   // std::size_t
#include <iostream>  // std::cout, std::endl
#include <memory>    // std::shared_ptr, std::make_shared
#include <string>    // std::string
#include <tuple>     // std::tie
#include <utility>   // std::move
#include <vector>    // std::vector

namespace plssvm {

/**
 * @example model_examples.cpp
 * @brief A few examples regarding the plssvm::model class.
 */

/**
 * @brief Implements a class encapsulating the result of a call to the SVM fit function. A model is used to predict the labels of a new data set.
 * @tparam U the type of the used labels (must be an arithmetic type or `std:string`; default: `int`)
 */
template <typename U = int>
class model {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<U, detail::supported_label_types>, "Illegal label type provided! See the 'label_type_list' in the type_list.hpp header for a list of the allowed types.");

    // plssvm::csvm needs the private constructor
    friend class csvm;

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
    explicit model(const std::string &filename);

    /**
     * @brief Save the model to a LIBSVM model file for later usage.
     * @param[in] filename the file to save the model to
     */
    void save(const std::string &filename) const;

    /**
     * @brief The number of support vectors used in this model.
     * @return the number of support vectors (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_support_vectors() const noexcept { return num_support_vectors_; }
    /**
     * @brief The number of features of the support vectors used in this model.
     * @return the number of features (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_features() const noexcept { return num_features_; }

    /**
     * @brief Return the SVM parameter that were used to learn this model.
     * @return the SVM parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] const parameter &get_params() const noexcept { return params_; }
    /**
     * @brief The support vectors representing the learned model.
     * @details The support vectors are of dimension `num_support_vectors()` x `num_features()`.
     * @return the support vectors (`[[nodiscard]]`)
     */
    [[nodiscard]] const soa_matrix<real_type> &support_vectors() const noexcept { return data_.data(); }

    /**
     * @brief Returns the labels of the support vectors.
     * @details If the labels are present, they can be retrieved as `std::vector` using: `dataset.labels()->%get()`.
     * @return the labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<label_type> &labels() const noexcept { return data_.labels()->get(); }
    /**
     * @brief Returns the number of classes in this model.
     * @details If the data set contains the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns `2`.
     *          It is the same as: `model.classes().size()`
     * @return the number of classes (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_classes() const noexcept { return data_.num_classes(); }
    /**
     * @brief Returns the classes of the support vectors.
     * @details If the support vectors contain the labels `std::vector<int>{ -1, 1, 1, -1, -1, 1 }`, this function returns the classes `{ -1, 1 }`.
     * @return all classes (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<label_type> classes() const { return data_.classes().value(); }

    /**
     * @brief The learned weights for the support vectors.
     * @details It is of size `num_classes() x num_support_vectors()`.
     * @return the weights (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<aos_matrix<real_type>> &weights() const noexcept {
        PLSSVM_ASSERT(alpha_ptr_ != nullptr, "The alpha_ptr may never be a nullptr!");
        return *alpha_ptr_;
    }
    /**
     * @brief The bias values for the different classes after learning.
     * @return the bias `rho` (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<real_type> &rho() const noexcept {
        PLSSVM_ASSERT(rho_ptr_ != nullptr, "The rho_ptr may never be a nullptr!");
        return *rho_ptr_;
    }
    /**
     * @brief Returns the multi-class classification strategy used to ft this model.
     * @return the multi-class classification strategy (`[[nodiscard]]`)
     */
    [[nodiscard]] classification_type get_classification_type() const noexcept { return classification_strategy_; }
    /**
     * @brief Returns the number of CG iterations to learn the classifiers used for this model.
     * @details Equal to `std::nullopt` if the model has been read from a model file.
     * @return the number of iterations (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::optional<std::vector<unsigned long long>> &num_iters() const noexcept { return num_iters_; }

  private:
    /**
     * @brief Create a new model using the SVM parameter @p params and the @p data.
     * @details Default initializes the weights, i.e., no weights have currently been learned.
     * @note This constructor may only be used in the befriended base C-SVM class!
     * @param[in] params the SVM parameters used to learn this model
     * @param[in] data the data used to learn this model
     * @param[in] classification_strategy the classification strategy used to fit this model
     */
    model(parameter params, data_set<label_type> data, classification_type classification_strategy);

    /// The SVM parameter used to learn this model.
    parameter params_{};
    /// The classification strategy (one vs. all or one vs. one) used to fit this model.
    classification_type classification_strategy_{};
    /// The data (support vectors + respective label) used to learn this model.
    data_set<label_type> data_{};
    /// The number of support vectors representing this model.
    size_type num_support_vectors_{ 0 };
    /// The number of features per support vector.
    size_type num_features_{ 0 };
    /// The number of iterations needed to fit this model.
    std::optional<std::vector<unsigned long long>> num_iters_{};

    /**
     * @brief The learned weights for each support vector.
     * @note Must be initialized to an empty vector instead of a `nullptr`.
     */
    std::shared_ptr<std::vector<aos_matrix<real_type>>> alpha_ptr_{ std::make_shared<std::vector<aos_matrix<real_type>>>() };

    /**
     * @brief For each class, holds the indices of all data points in the support vectors.
     * @note Must be initialized to an empty vector instead of a `nullptr`.
     */
    std::shared_ptr<std::vector<std::vector<std::size_t>>> index_sets_ptr_{ std::make_shared<std::vector<std::vector<std::size_t>>>() };

    /**
     * @brief The bias after learning this model.
     * @note Must be initialized to an empty vector instead of a `nullptr`.
     */
    std::shared_ptr<std::vector<real_type>> rho_ptr_{ std::make_shared<std::vector<real_type>>() };

    /**
     * @brief A vector used to speedup the prediction in case of the linear kernel function.
     * @details Will be reused by subsequent calls to `plssvm::csvm::fit`/`plssvm::csvm::score` with the same `plssvm::model`.
     * @note Must be initialized to an empty vector instead of a `nullptr` in order to be passable as const reference.
     */
    std::shared_ptr<soa_matrix<real_type>> w_ptr_{ std::make_shared<soa_matrix<real_type>>() };
};

template <typename U>
model<U>::model(parameter params, data_set<label_type> data, const classification_type classification_strategy) :
    params_{ std::move(params) }, classification_strategy_{ classification_strategy }, data_{ std::move(data) }, num_support_vectors_{ data_.num_data_points() }, num_features_{ data_.num_features() } {}

template <typename U>
model<U>::model(const std::string &filename) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // parse the libsvm model header
    std::vector<label_type> labels{};
    std::vector<label_type> unique_labels{};
    std::vector<std::size_t> num_sv_per_class{};
    std::size_t num_header_lines{};
    std::tie(params_, *rho_ptr_, labels, unique_labels, num_sv_per_class, num_header_lines) = detail::io::parse_libsvm_model_header<label_type>(reader.lines());

    // create empty support vectors and alpha vector
    soa_matrix<real_type> support_vectors{};

    // parse libsvm model data
    std::tie(num_support_vectors_, num_features_, support_vectors, *alpha_ptr_, classification_strategy_) = detail::io::parse_libsvm_model_data(reader, num_sv_per_class, num_header_lines);

    switch (classification_strategy_) {
        case classification_type::oaa:
            // empty index set for the OAA classification
            index_sets_ptr_ = std::make_shared<std::vector<std::vector<std::size_t>>>();
            break;
        case classification_type::oao: {
            // fill index_sets -> support vectors are sorted!
            index_sets_ptr_ = std::make_shared<std::vector<std::vector<std::size_t>>>(unique_labels.size());
            std::size_t running_idx{ 0 };
            for (std::size_t i = 0; i < num_sv_per_class.size(); ++i) {
                (*index_sets_ptr_)[i] = std::vector<std::size_t>(num_sv_per_class[i]);
                std::iota((*index_sets_ptr_)[i].begin(), (*index_sets_ptr_)[i].end(), running_idx);
                running_idx += num_sv_per_class[i];
            }
        } break;
    }

    // create data set
    PLSSVM_ASSERT(support_vectors.num_rows() == labels.size(), "Number of labels ({}) must match the number of data points ({})!", labels.size(), support_vectors.num_rows());
    const verbosity_level old_verbosity = verbosity;
    verbosity = verbosity_level::quiet;
    data_ = data_set<label_type>{};
    data_.num_data_points_ = support_vectors.num_rows();
    data_.num_features_ = support_vectors.num_cols();
    (*data_.data_ptr_) = std::move(support_vectors);
    data_.labels_ptr_ = std::make_shared<typename decltype(data_.labels_ptr_)::element_type>(std::move(labels));  // prevent multiple calls to "create_mapping"
    data_.create_mapping(unique_labels);
    verbosity = old_verbosity;

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Read {} support vectors with {} features and {} classes using {} classification in {} using the libsvm model parser from file '{}'.\n\n",
                detail::tracking_entry{ "model_read", "num_support_vectors", num_support_vectors_ },
                detail::tracking_entry{ "model_read", "num_features", num_features_ },
                detail::tracking_entry{ "model_read", "num_classes", this->num_classes() },
                classification_type_to_full_string(classification_strategy_),
                detail::tracking_entry{ "model_read", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking_entry{ "model_read", "filename", filename });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "model_read", "rho", *rho_ptr_ }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "model_read", "classification_type", classification_strategy_ }));
}

template <typename U>
void model<U>::save(const std::string &filename) const {
    PLSSVM_ASSERT(rho_ptr_ != nullptr, "The rho_ptr may never be a nullptr!");
    PLSSVM_ASSERT(alpha_ptr_ != nullptr, "The alpha_ptr may never be a nullptr!");
    PLSSVM_ASSERT(index_sets_ptr_ != nullptr, "The index_sets_ptr may never be a nullptr!");

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // save model file header and support vectors
    detail::io::write_libsvm_model_data(filename, params_, classification_strategy_, *rho_ptr_, *alpha_ptr_, *index_sets_ptr_, data_);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Write {} support vectors with {} features and {} classes using {} classification in {} to the libsvm model file '{}'.\n",
                detail::tracking_entry{ "model_write", "num_support_vectors", num_support_vectors_ },
                detail::tracking_entry{ "model_write", "num_features", num_features_ },
                detail::tracking_entry{ "model_write", "num_classes", this->num_classes() },
                classification_type_to_full_string(classification_strategy_),
                detail::tracking_entry{ "model_write", "time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) },
                detail::tracking_entry{ "model_write", "filename", filename });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "model_write", "rho", *rho_ptr_ }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "model_write", "classification_type", classification_strategy_ }));
}

}  // namespace plssvm

#endif  // PLSSVM_MODEL_HPP_

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

#ifndef PLSSVM_MODEL_MODEL_HPP_
#define PLSSVM_MODEL_MODEL_HPP_
#pragma once

#include "plssvm/constants.hpp"          // plssvm::real_type
#include "plssvm/data_set/data_set.hpp"  // plssvm::data_set, plssvm::optional_ref
#include "plssvm/detail/assert.hpp"      // PLSSVM_ASSERT
#include "plssvm/matrix.hpp"             // plssvm::soa_matrix, plssvm::aos_matrix
#include "plssvm/parameter.hpp"          // plssvm::parameter

#include <cstddef>   // std::size_t
#include <memory>    // std::shared_ptr, std::make_shared
#include <optional>  // std::optional
#include <string>    // std::string
#include <utility>   // std::move
#include <vector>    // std::vector

namespace plssvm {

/**
 * @brief Implements a class encapsulating the result of a call to the SVM fit function. A model is used to predict the labels of a new data set.
 * @tparam U the type of the used labels (must be an arithmetic type or `std:string`; default: `int`)
 */
template <typename U>
class model {
  public:
    /// The type of the labels: any arithmetic type or `std::string`.
    using label_type = U;
    /// The unsigned size type.
    using size_type = std::size_t;

    /**
     * @brief Default copy constructor.
     */
    model(const model &) = default;
    /**
     * @brief Default move constructor.
     */
    model(model &&) noexcept = default;
    /**
     * @brief Default copy assignment operator.
     * @return `*this`
     */
    model &operator=(const model &) = default;
    /**
     * @brief Default move assignment operator.
     * @return `*this`
     */
    model &operator=(model &&) noexcept = default;

    /**
     * @brief Virtual destructor to allow derived classes to clean up properly.
     */
    virtual ~model() = default;

    /**
     * @brief Save the model to a LIBSVM model file for later usage.
     * @param[in] filename the file to save the model to
     */
    virtual void save(const std::string &filename) const = 0;

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
    [[nodiscard]] const soa_matrix<real_type> &support_vectors() const noexcept { return data_->data(); }

    /**
     * @brief Returns an optional reference to the labels of the support vectors.
     * @details If the labels are present, they can be retrieved as `std::vector` using: `dataset.labels()->%get()`.
     * @return the labels (`[[nodiscard]]`)
     */
    [[nodiscard]] optional_ref<const std::vector<label_type>> labels() const noexcept { return data_->labels(); }

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
     * @brief Returns the number of CG iterations to learn the classifiers used for this model.
     * @details Equal to `std::nullopt` if the model has been read from a model file.
     * @return the number of iterations (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::optional<std::vector<unsigned long long>> &num_iters() const noexcept { return num_iters_; }

  protected:
    /**
     * @brief Default construct an empty model.
     */
    model() = default;
    /**
     * @brief Create a new model using the SVM parameter @p params and the support vectors @p data.
     * @param[in] params the SVM parameter
     * @param[in] data the support vectors
     */
    model(parameter params, std::shared_ptr<data_set<label_type>> data);

    /// The SVM parameter used to learn this model.
    parameter params_{};
    /// The data (support vectors + respective label) used to learn this model.
    std::shared_ptr<data_set<label_type>> data_{};
    /// The number of support vectors representing this model.
    size_type num_support_vectors_{ 0 };
    /// The number of features per support vector.
    size_type num_features_{ 0 };
    /// The number of iterations needed to fit this model.
    std::optional<std::vector<unsigned long long>> num_iters_{};

    /**
     * @brief The learned weights for each support vector.
     * @details For one vs. all the vector contains a single matrix representing all weights.
     *          For one vs. one the vector contains one weight matrix for each binary classification pair.
     * @note Must be initialized to an empty vector instead of a `nullptr`.
     */
    std::shared_ptr<std::vector<aos_matrix<real_type>>> alpha_ptr_{ std::make_shared<std::vector<aos_matrix<real_type>>>() };

    /**
     * @brief The bias after learning this model.
     * @details The number of entries depends on the used classification type.
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
model<U>::model(parameter params, std::shared_ptr<data_set<label_type>> data) :
    params_{ std::move(params) },
    data_{ std::move(data) },
    num_support_vectors_{ data_->num_data_points() },
    num_features_{ data_->num_features() } { }

}  // namespace plssvm

#endif  // PLSSVM_MODEL_MODEL_HPP_

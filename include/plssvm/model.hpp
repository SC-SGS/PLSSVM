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

#include "plssvm/constants.hpp"                       // plssvm::verbose
#include "plssvm/data_set.hpp"                        // plssvm::data_set
#include "plssvm/detail/assert.hpp"                   // PLSSVM_ASSERT
#include "plssvm/detail/io/libsvm_model_parsing.hpp"  // plssvm::detail::io::{parse_libsvm_model_header, write_libsvm_model_data}
#include "plssvm/detail/io/libsvm_parsing.hpp"        // plssvm::detail::io::parse_libsvm_data
#include "plssvm/parameter.hpp"                       // plssvm::parameter

#include "fmt/chrono.h"                               // format std::chrono types using fmt
#include "fmt/core.h"                                 // fmt::format

#include <chrono>                                     // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>                                    // std::size_t
#include <iostream>                                   // std::cout, std::endl
#include <memory>                                     // std::shared_ptr, std::make_shared
#include <string>                                     // std::string
#include <tuple>                                      // std::tie
#include <type_traits>                                // std::is_same_v, std::is_arithmetic_v
#include <utility>                                    // std::move
#include <vector>                                     // std::vector

namespace plssvm {

/**
 * @example model_examples.cpp
 * @brief A few examples regarding the plssvm::model class.
 */

/**
 * @brief Implements a class encapsulating the result of a call to the SVM fit function. A model is used to predict the labels of a new data set.
 * @tparam T the floating point type of the data (must either be `float` or `double`)
 * @tparam U the type of the used labels (must be an arithmetic type or `std:string`; default: `int`)
 */
template <typename T, typename U = int>
class model {
    // make sure only valid template types are used
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The first template type can only be 'float' or 'double'!");
    static_assert(std::is_arithmetic_v<U> || std::is_same_v<U, std::string>, "The second template type can only be an arithmetic type or 'std::string'!");

    // plssvm::csvm needs the private constructor
    friend class csvm;

  public:
    /// The type of the data points: either `float` or `double`.
    using real_type = T;
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
    [[nodiscard]] const std::vector<std::vector<real_type>> &support_vectors() const noexcept { return data_.data(); }
    /**
     * @brief The learned weights for the support vectors.
     * @details It is of size `num_support_vectors()`.
     * @return the weights (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<real_type> &weights() const noexcept {
        PLSSVM_ASSERT(alpha_ptr_ != nullptr, "The alpha_ptr may never be a nullptr!");
        return *alpha_ptr_;
    }
    /**
     * @brief The bias value after learning.
     * @return the bias `rho` (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type rho() const noexcept { return rho_; }

  private:
    /**
     * @brief Create a new model using the SVM parameter @p params and the @p data.
     * @details Default initializes the weights, i.e., no weights have currently been learned.
     * @note This constructor may only be used in the befriended base C-SVM class!
     * @param[in] params the SVM parameters used to learn this model
     * @param[in] data the data used to learn this model
     */
    model(parameter params, data_set<real_type, label_type> data);

    /// The SVM parameter used to learn this model.
    parameter params_{};
    /// The data (support vectors + respective label) used to learn this model.
    data_set<real_type, label_type> data_{};
    /// The number of support vectors representing this model.
    size_type num_support_vectors_{ 0 };
    /// The number of features per support vector.
    size_type num_features_{ 0 };

    /// The learned weights for each support vector.
    std::shared_ptr<std::vector<real_type>> alpha_ptr_{ nullptr };
    /// The bias after learning this model.
    real_type rho_{ 0.0 };

    /**
     * @brief A vector used to speedup the prediction in case of the linear kernel function.
     * @details Will be reused by subsequent calls to `plssvm::csvm::fit`/`plssvm::csvm::score` with the same `plssvm::model`.
     * @note Must be initialized to an empty vector instead of a `nullptr` in order to be passable as const reference.
     */
    std::shared_ptr<std::vector<real_type>> w_{ std::make_shared<std::vector<real_type>>() };
};

template <typename T, typename U>
model<T, U>::model(const std::string &filename) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // open the file
    detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // parse the libsvm model header
    std::vector<label_type> labels{};
    std::size_t num_header_lines{};
    std::tie(params_, rho_, labels, num_header_lines) = detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // create empty support vectors and alpha vector
    std::vector<std::vector<real_type>> support_vectors;
    std::vector<real_type> alphas;

    // parse libsvm model data
    std::tie(num_support_vectors_, num_features_, support_vectors, alphas) = detail::io::parse_libsvm_data<real_type, real_type>(reader, num_header_lines);

    // create data set
    data_ = data_set<real_type, label_type>{ std::move(support_vectors), std::move(labels) };
    alpha_ptr_ = std::make_shared<decltype(alphas)>(std::move(alphas));

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << fmt::format("Read {} support vectors with {} features in {} using the libsvm model parser from file '{}'.\n",
                                 num_support_vectors_,
                                 num_features_,
                                 std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                                 filename)
                  << std::endl;
    }
}

template <typename T, typename U>
model<T, U>::model(parameter params, data_set<real_type, label_type> data) :
    params_{ std::move(params) }, data_{ std::move(data) }, num_support_vectors_{ data_.num_data_points() }, num_features_{ data_.num_features() }, alpha_ptr_{ std::make_shared<std::vector<real_type>>(data_.num_data_points()) } {}

template <typename T, typename U>
void model<T, U>::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // save model file header and support vectors
    detail::io::write_libsvm_model_data(filename, params_, rho_, *alpha_ptr_, data_);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << fmt::format("Write {} support vectors with {} features in {} to the libsvm model file '{}'.",
                                 num_support_vectors_,
                                 num_features_,
                                 std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                                 filename)
                  << std::endl;
    }
}

}  // namespace plssvm

#endif  // PLSSVM_MODEL_HPP_

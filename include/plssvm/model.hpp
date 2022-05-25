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

#pragma once

#include "plssvm/constants.hpp"
#include "plssvm/parameter.hpp"
#include "plssvm/data_set.hpp"
#include "plssvm/detail/io/libsvm_parsing.hpp"
#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "fmt/core.h"
#include "fmt/os.h"

#include <chrono>
#include <memory>       // std::shared_ptr, std::make_shared
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector
#include <sstream>
#include <numeric>

namespace plssvm {

template <typename T, typename U = int>
class model {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The first template type can only be 'float' or 'double'!");
    static_assert(std::is_arithmetic_v<U> || std::is_same_v<U, std::string>, "The second template type can only be an arithmetic type or 'std::string'!");
    // because std::vector<bool> is evil
    static_assert(!std::is_same_v<U, bool>, "The second template type must NOT be 'bool'!");

    // plssvm::csvm needs the constructor
    template <typename>
    friend class csvm;

  public:
    using real_type = T;
    using label_type = U;
    using size_type = std::size_t;

    explicit model(const std::string& filename);

    void save(const std::string& filename) const;

    [[nodiscard]] size_type num_support_vectors() const noexcept { return num_support_vectors_; }
    [[nodiscard]] size_type num_features() const noexcept { return num_features_; }

  private:
    model(parameter<real_type> params, data_set<real_type, label_type> data);

    parameter<real_type> params_{};

    data_set<real_type, label_type> data_{}; // support vectors + labels
    std::shared_ptr<std::vector<real_type>> alpha_ptr_{ nullptr };

    // used to speedup prediction in case of the linear kernel function
    std::shared_ptr<std::vector<real_type>> w_{ nullptr };

    real_type rho_{ 0.0 };

    size_type num_support_vectors_{ 0 };
    size_type num_features_{ 0 };
};

/******************************************************************************
 *                                Constructors                                *
 ******************************************************************************/
 // TODO: restructure?
template <typename T, typename U>
model<T, U>::model(const std::string &filename) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    detail::io::file_reader f{ filename, '#' };

    // parse libsvm header
    auto [header, data_labels] = detail::io::read_libsvm_model_header<real_type, label_type, size_type>(f, params_, rho_, num_support_vectors_);

    // create support vectors and alpha pointer
    num_features_ = detail::io::parse_libsvm_num_features(f, num_support_vectors_, header + 1);
    std::vector<std::vector<real_type>> support_vectors(num_support_vectors_, std::vector<real_type>(num_features_));
    alpha_ptr_ = std::make_shared<std::vector<real_type>>(num_support_vectors_);

    // parse libsvm model data
    detail::io::read_libsvm_data(f, header + 1, support_vectors, *alpha_ptr_);

    // create data set
    data_ = data_set<real_type, label_type>{ std::move(support_vectors), std::move(data_labels) };

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Read {} support vectors with {} features in {} using the libsvm model parser from file '{}'.\n",
                   num_support_vectors_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   filename);
    }
}

template <typename T, typename U>
model<T, U>::model(parameter<real_type> params, data_set<real_type, label_type> data)
    : params_{ std::move(params) }, data_{ std::move(data) }, alpha_ptr_{ std::make_shared<std::vector<real_type>>(data_.num_data_points()) },
      num_support_vectors_{ data_.num_data_points() }, num_features_{ data_.num_features() } {}

/******************************************************************************
 *                                 Save Model                                 *
 ******************************************************************************/
template <typename T, typename U>
void model<T, U>::save(const std::string &filename) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    fmt::ostream out = fmt::output_file(filename);

    // save model file header
    const std::vector<label_type> label_order = detail::io::write_libsvm_model_header(out, params_, rho_, data_);

    // save model file support vectors
    detail::io::write_libsvm_model_data(out, data_.data(), *alpha_ptr_, data_.labels().value().get(), label_order);

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Write {} support vectors with {} features in {} to the libsvm model file '{}'.\n",
                   num_support_vectors_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   filename);
    }
}

}
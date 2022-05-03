/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a data set class encapsulating all data points and potential features.
 */

#pragma once

#include "plssvm/file_format_types.hpp"  // plssvm::file_format_type

#include <cstddef>      // std::size_t
#include <functional>   // std::reference_wrapper, std::cref
#include <memory>       // std::shared_ptr
#include <optional>     // std::optional, std::make_optional, std::nullopt
#include <string>       // std::string
#include <vector>       // std::vector

#include <variant>      // std::variant

namespace plssvm {

template <typename T>
class data_set {
    using data_matrix_type = std::vector<std::vector<T>>;
    using label_vector_type = std::vector<T>;

    std::map<T, std::string> mapping_;

  public:
    using real_type = T;
    using label_type = T;
    using size_type = std::size_t;

    data_set();

    explicit data_set(const std::string& filename);
    data_set(const std::string& filename, file_format_type format);

    explicit data_set(data_matrix_type &&X);
    data_set(data_matrix_type &&X, std::vector<real_type> &&y);

    data_set(data_matrix_type &&X, std::vector<int> &&x);
    data_set(data_matrix_type &&X, std::vector<std::string> &&x);


    // save the data set in the given format
    void save_data_set(const std::string& filename, file_format_type format) const;

    // scale data features to be in range [-1, +1]
    void scale();
    // scale data features to be in range [lower, upper]
    void scale(real_type lower, real_type upper);

    [[nodiscard]] const data_matrix_type& data() const noexcept { return *X_ptr_; }
    [[nodiscard]] std::optional<std::reference_wrapper<const label_vector_type>> labels() const noexcept {
        if (this->has_labels()) {
            return std::make_optional(std::cref(*y_ptr_));
        } else {
            return std::nullopt;
        }
    }
    [[nodiscard]] bool has_labels() const noexcept { return y_ptr_ != nullptr; }

    [[nodiscard]] size_type num_data_points() const noexcept { return num_data_points_; }
    [[nodiscard]] size_type num_features() const noexcept { return num_features_; }

  private:
    void write_libsvm_file(const std::string& filename) const;
    void write_arff_file(const std::string& filename) const;
    void read_file(const std::string& filename, file_format_type format);
    void read_libsvm_file(const std::string& filename);
    void read_arff_file(const std::string& filename);

    std::shared_ptr<data_matrix_type> X_ptr_{ nullptr };
    std::shared_ptr<label_vector_type> y_ptr_{ nullptr };

    size_type num_data_points_{ 0 };
    size_type num_features_{ 0 };
};

extern template class data_set<float>;
extern template class data_set<double>;

}
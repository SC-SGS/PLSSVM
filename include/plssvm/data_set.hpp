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

#include "plssvm/constants.hpp"                 // plssvm::verbose
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::exception
#include "plssvm/file_format_types.hpp"         // plssvm::file_format_type
#include "plssvm/detail/io/arff_parsing.hpp"    // plssvm::detail::io::{read_libsvm_data, write_libsvm_data}
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_parsing.hpp"  // plssvm::detail::io::{read_arff_header, read_arff_data, write_arff_header, write_arff_data}
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::ends_with

#include "fmt/core.h"    // fmt::format, fmt::print
#include "fmt/chrono.h"  // directly output std::chrono times via fmt
#include "fmt/os.h"      // fmt::ostream
#include "fmt/ostream.h" // directly output objects with operator<< overload via fmt

#include <algorithm>   // std::all_of, std::max, std::min
#include <chrono>      // std::chrono::{time_point, steady_clock, duration_cast, millisecond}
#include <cstddef>     // std::size_t
#include <functional>  // std::reference_wrapper, std::cref
#include <limits>      // std::numeric_limits
#include <map>         // std::map
#include <memory>      // std::shared_ptr, std::make_shared
#include <optional>    // std::optional, std::make_optional, std::nullopt
#include <set>         // std::set
#include <string>      // std::string
#include <tuple>       // std::tie
#include <utility>     // std::move, std::pair
#include <vector>      // std::vector


namespace plssvm {

template <typename T>
using optional_ref = std::optional<std::reference_wrapper<T>>;

template <typename T, typename U = int>
class data_set {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The first template type can only be 'float' or 'double'!");
    static_assert(std::is_arithmetic_v<U> || std::is_same_v<U, std::string>, "The second template type can only be an arithmetic type or 'std::string'!");
    // because std::vector<bool> is evil
    static_assert(!std::is_same_v<U, bool>, "The second template type must NOT be 'bool'!");

  public:
    using real_type = T;
    using label_type = U;
    using size_type = std::size_t;

    data_set();

    explicit data_set(const std::string& filename, std::optional<std::pair<real_type, real_type>> scaling = std::nullopt);
    data_set(const std::string& filename, file_format_type format, std::optional<std::pair<real_type, real_type>> scaling = std::nullopt);

    explicit data_set(std::vector<std::vector<real_type>> &&X, std::optional<std::pair<real_type, real_type>> scaling = std::nullopt);
    data_set(std::vector<std::vector<real_type>> &&X, std::vector<label_type> &&y, std::optional<std::pair<real_type, real_type>> scaling = std::nullopt);


    // save the data set in the given format
    void save_data_set(const std::string& filename, file_format_type format) const;

    [[nodiscard]] const std::vector<std::vector<real_type>>& data() const noexcept { return *X_ptr_; }
    [[nodiscard]] bool has_labels() const noexcept { return y_ptr_ != nullptr; }
    [[nodiscard]] optional_ref<const std::vector<real_type>> mapped_labels() const noexcept;
    [[nodiscard]] optional_ref<const std::vector<label_type>> labels() noexcept;

    [[nodiscard]] label_type label_from_mapped_value(real_type val) const noexcept;

    [[nodiscard]] size_type num_labels() const noexcept { return mapping_.size(); }
    [[nodiscard]] size_type num_data_points() const noexcept { return num_data_points_; }
    [[nodiscard]] size_type num_features() const noexcept { return num_features_; }

  private:
    void create_mapping();
    void scale(std::pair<real_type, real_type> scaling);

    void write_libsvm_file(const std::string& filename) const;
    void write_arff_file(const std::string& filename) const;
    void read_file(const std::string& filename, file_format_type format);
    void read_libsvm_file(const std::string& filename);
    void read_arff_file(const std::string& filename);

    std::shared_ptr<std::vector<std::vector<real_type>>> X_ptr_{ nullptr };
    std::shared_ptr<std::vector<real_type>> y_ptr_{ nullptr };
    std::shared_ptr<std::vector<label_type>> labels_ptr_{ nullptr };

    size_type num_data_points_{ 0 };
    size_type num_features_{ 0 };

    std::map<real_type, label_type> mapping_{};
};


/******************************************************************************
 *                                Constructors                                *
 ******************************************************************************/
template <typename T, typename U>
data_set<T, U>::data_set() : X_ptr_{ std::make_shared<std::vector<std::vector<real_type>>>() } { }

template <typename T, typename U>
data_set<T, U>::data_set(const std::string& filename, const std::optional<std::pair<real_type, real_type>> scaling) {
    // read data set from file
    this->read_file(filename, detail::ends_with(filename, ".arff") ? file_format_type::arff : file_format_type::libsvm);
    // scale data set
    if (scaling.has_value()) {
       this->scale(scaling.value());
    }
}

template <typename T, typename U>
data_set<T, U>::data_set(const std::string &filename, const file_format_type format, const std::optional<std::pair<real_type, real_type>> scaling) {
    // read data set from file
    this->read_file(filename, format);
    // scale data set
    if (scaling.has_value()) {
        this->scale(scaling.value());
    }
}

template <typename T, typename U>
data_set<T, U>::data_set(std::vector<std::vector<real_type>> &&X, const std::optional<std::pair<real_type, real_type>> scaling) : X_ptr_{ std::make_shared<std::vector<std::vector<real_type>>>(std::move(X)) } {
    if (X_ptr_->empty()) {
        throw exception("Data vector is empty!");
    } else if (!std::all_of(X_ptr_->begin(), X_ptr_->end(), [&](const std::vector<real_type>& point) { return point.size() == X_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (X_ptr_->front().size() == 0) {
        throw exception{ "No features provided for the data points!" };
    }

    num_data_points_ = X_ptr_->size();
    num_features_ = X_ptr_->front().size();

    // scale data set
    if (scaling.has_value()) {
        this->scale(scaling.value());
    }
}

template <typename T, typename U>
data_set<T, U>::data_set(std::vector<std::vector<real_type>> &&X, std::vector<label_type> &&y, const std::optional<std::pair<real_type, real_type>> scaling) : X_ptr_{ std::make_shared<std::vector<std::vector<real_type>>>(std::move(X)) }, labels_ptr_{ std::make_shared<std::vector<label_type>>(std::move(y)) } {
    if (X_ptr_->empty()) {
        throw exception("Data vector is empty!");
    } else if (!std::all_of(X_ptr_->begin(), X_ptr_->end(), [&](const std::vector<real_type>& point) { return point.size() == X_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (X_ptr_->front().size() == 0) {
        throw exception{ "No features provided for the data points!" };
    } else if (X_ptr_->size() != labels_ptr_->size()) {
        throw exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", labels_ptr_->size(), X_ptr_->size()) };
    }

    // create mapping from labels
    this->create_mapping();

    num_data_points_ = X_ptr_->size();
    num_features_ = X_ptr_->front().size();

    // scale data set
    if (scaling.has_value()) {
        this->scale(scaling.value());
    }
}


/******************************************************************************
 *                                Save Data Set                               *
 ******************************************************************************/
template <typename T, typename U>
void data_set<T, U>::save_data_set(const std::string &filename, const file_format_type format) const {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // save the data set
    switch (format) {
        case file_format_type::libsvm:
            this->write_libsvm_file(filename);
            break;
        case file_format_type::arff:
            this->write_arff_file(filename);
            break;
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Write {} data points with {} features in {} to the {} file '{}'.\n",
                   num_data_points_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   format,
                   filename);
    }
}


/******************************************************************************
 *                                   Getter                                   *
 ******************************************************************************/
template <typename T, typename U>
auto data_set<T, U>::mapped_labels() const noexcept -> optional_ref<const std::vector<real_type>> {
    if (this->has_labels()) {
        return std::make_optional(std::cref(*y_ptr_));
    } else {
        return std::nullopt;
    }
}

template <typename T, typename U>
auto data_set<T, U>::labels() noexcept -> optional_ref<const std::vector<label_type>> {
    if (this->has_labels()) {
        return std::make_optional(std::cref(*labels_ptr_));
    } else {
        return std::nullopt;
    }
}

template <typename T, typename U>
auto data_set<T, U>::label_from_mapped_value(const real_type val) const noexcept -> label_type {
    if (mapping_.count(val) == 0) {
        throw exception{ "Illegal mapped value: {}!" };
    }
    return mapping_[val];
}


////////////////////////////////////////////////////////////////////////////////
////                        PRIVATE MEMBER FUNCTIONS                        ////
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
void data_set<T, U>::create_mapping() {
    PLSSVM_ASSERT(labels_ptr_ != nullptr, "Can't create mapping if no labels are provided!");

    // get unique labels
    const std::set<U> unique_labels(labels_ptr_->begin(), labels_ptr_->end());
    // only binary classification allowed as of now
    if (unique_labels.size() != 2) {
        throw exception{fmt::format("Currently only binary classification is supported, but {} different label where given!", unique_labels.size())};
    }
    // map binary labels to -1 and 1
    mapping_[-1] = *unique_labels.begin();
    mapping_[1] = *(++unique_labels.begin());

    // convert input labels to now mapped values
    std::vector<real_type> tmp(labels_ptr_->size());
    #pragma omp parallel for default(none) shared(tmp, labels_ptr_, mapping_)
    for (typename std::vector<real_type>::size_type i = 0; i < tmp.size(); ++i) {
        tmp[i] = std::find_if(mapping_.begin(), mapping_.end(), [&](const auto &kv) { return kv.second == (*labels_ptr_)[i]; })->first;
    }
    y_ptr_ = std::make_shared<std::vector<real_type>>(std::move(tmp));
}

template <typename T, typename U>
void data_set<T, U>::scale(const std::pair<real_type, real_type> scaling) {
    // unpack pair
    const real_type lower = scaling.first;
    const real_type upper = scaling.second;
    if (lower >= upper) {
        throw plssvm::exception(fmt::format("Illegal interval specification: lower ({}) < upper ({}).", lower, upper));
    }

    real_type min_entry = std::numeric_limits<real_type>::max();
    real_type max_entry = std::numeric_limits<real_type>::lowest();

    #pragma omp parallel for collapse(2) reduction(min:min_entry) reduction(max:max_entry) default(none) shared(X_ptr_) firstprivate(num_data_points_, num_features_)
    for (size_type row = 0; row < num_data_points_; ++row) {
        for (size_type col = 0; col < num_features_; ++col) {
            min_entry = std::min(min_entry, (*X_ptr_)[row][col]);
            max_entry = std::max(max_entry, (*X_ptr_)[row][col]);
        }
    }

    #pragma omp parallel for collapse(2) default(none) shared(X_ptr_) firstprivate(upper, lower, max_entry, min_entry, num_data_points_, num_features_)
    for (size_type row = 0; row < num_data_points_; ++row) {
        for (size_type col = 0; col < num_features_; ++col) {
            (*X_ptr_)[row][col] = (upper - lower) / (max_entry - min_entry) * ((*X_ptr_)[row][col] - max_entry) + upper;
        }
    }
}


/******************************************************************************
 *                                 Write Files                                *
 ******************************************************************************/
template <typename T, typename U>
void data_set<T, U>::write_libsvm_file(const std::string& filename) const {
    fmt::ostream out = fmt::output_file(filename);
    // write data
    if (this->has_labels()) {
        detail::io::write_libsvm_data<real_type, label_type, true>(out, X_ptr_, labels_ptr_);
    } else {
        detail::io::write_libsvm_data<real_type, label_type, false>(out, X_ptr_);
    }
}

template <typename T, typename U>
void data_set<T, U>::write_arff_file(const std::string &filename) const {
    fmt::ostream out = fmt::output_file(filename);

    // write header information
    detail::io::write_arff_header(out, num_features_, this->has_labels());

    // write data
    if (this->has_labels()) {
        detail::io::write_arff_data(out, X_ptr_, y_ptr_);
    } else {
        detail::io::write_arff_data(out, X_ptr_);
    }
}


/******************************************************************************
 *                                 Read Files                                 *
 ******************************************************************************/
template <typename T, typename U>
void data_set<T, U>::read_file(const std::string &filename, file_format_type format) {
    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // parse the given file
    switch (format) {
        case file_format_type::libsvm:
            read_libsvm_file(filename);
            break;
        case file_format_type::arff:
            read_arff_file(filename);
            break;
    }

    // create label mapping
    if (this->has_labels()) {
        this->create_mapping();
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Read {} data points with {} features in {} using the {} parser from file '{}'.\n",
                   num_data_points_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   format,
                   filename);
    }
}

template <typename T, typename U>
void data_set<T, U>::read_libsvm_file(const std::string &filename) {
    detail::io::file_reader f{ filename, '#' };

    // parse sizes
    num_data_points_ = f.num_lines();
    num_features_ = detail::io::parse_libsvm_num_features(f, f.num_lines(), 0);

    X_ptr_ = std::make_shared<std::vector<std::vector<real_type>>>(num_data_points_, std::vector<real_type>(num_features_));
    labels_ptr_ = std::make_shared<std::vector<label_type>>(num_data_points_);

    // parse file
    const bool has_label = detail::io::read_libsvm_data(f, 0, X_ptr_, labels_ptr_);

    // update shared pointer
    if (!has_label) {
        labels_ptr_ = nullptr;
    }
}

template <typename T, typename U>
void data_set<T,U>::read_arff_file(const std::string &filename) {
    detail::io::file_reader f{ filename, '%' };

    // parse arff header, structured binding
    size_type header = 0;
    size_type max_size = 0;
    bool has_label = false;
    std::tie(header, max_size, has_label) = detail::io::read_arff_header(f);

    num_data_points_ = f.num_lines() - (header + 1);
    num_features_ = has_label ? max_size - 1 : max_size;

    X_ptr_ = std::make_shared<std::vector<std::vector<real_type>>>(num_features_, std::vector<real_type>(num_features_));
    labels_ptr_ = std::make_shared<std::vector<label_type>>(num_data_points_);

    // parse file
    detail::io::read_arff_data(f, header, num_features_, max_size, has_label, X_ptr_, labels_ptr_);

    // update shared pointer
    if (!has_label) {
        labels_ptr_ = nullptr;
    }
}

}
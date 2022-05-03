/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"                 // plssvm::verbose
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::exception
#include "plssvm/file_format_types.hpp"         // plssvm::file_format_type
#include "plssvm/detail/io/arff_parsing.hpp"    // plssvm::detail::io::{read_libsvm_data, write_libsvm_data}
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_parsing.hpp"  // plssvm::detail::io::{read_arff_header, read_arff_data, write_arff_header, write_arff_data}
#include "plssvm/detail/operators.hpp"          // plssvm::operator::sign
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::ends_with

#include "fmt/core.h"    // fmt::format, fmt::print
#include "fmt/chrono.h"  // directly output std::chrono times via fmt
#include "fmt/os.h"      // fmt::ostream
#include "fmt/ostream.h" // directly output objects with operator<< overload via fmt

#include <algorithm>  // std::all_of, std::max, std::min
#include <chrono>     // std::chrono::{time_point, steady_clock, duration_cast, millisecond}
#include <limits>     // std::numeric_limits
#include <memory>     // std::shared_ptr, std::make_shared
#include <string>     // std::string
#include <tuple>      // std::tie
#include <utility>    // std::move
#include <vector>     // std::vector


namespace plssvm {

template <typename T>
data_set<T>::data_set() : X_ptr_{ std::make_shared<data_matrix_type>() }, y_ptr_{ nullptr }, num_data_points_{ 0 }, num_features_{ 0 } { }

template <typename T>
data_set<T>::data_set(const std::string& filename) {
    this->read_file(filename, detail::ends_with(filename, ".arff") ? file_format_type::arff : file_format_type::libsvm);
}

template <typename T>
data_set<T>::data_set(const std::string &filename, const file_format_type format) {
    this->read_file(filename, format);
}

template <typename T>
data_set<T>::data_set(data_matrix_type &&X) : X_ptr_{ std::make_shared<data_matrix_type>(std::move(X)) }, y_ptr_{ nullptr } {
    if (X_ptr_->empty()) {
        throw exception("Data vector is empty!");
    } else if (!std::all_of(X_ptr_->begin(), X_ptr_->end(), [&](const std::vector<real_type>& point) { return point.size() == X_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (X_ptr_->front().size() == 0) {
        throw exception{ "No features provided for the data points!" };
    }

    num_data_points_ = X_ptr_->size();
    num_features_ = X_ptr_->front().size();
}

template <typename T>
data_set<T>::data_set(data_matrix_type &&X, label_vector_type &&y) : X_ptr_{ std::make_shared<data_matrix_type>(std::move(X)) }, y_ptr_{ std::make_shared<label_vector_type>(std::move(y)) } {
    if (X_ptr_->empty()) {
        throw exception("Data vector is empty!");
    } else if (!std::all_of(X_ptr_->begin(), X_ptr_->end(), [&](const std::vector<real_type>& point) { return point.size() == X_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (X_ptr_->front().size() == 0) {
        throw exception{ "No features provided for the data points!" };
    } else if (X_ptr_->size() != y_ptr_->size()) {
        throw exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", y_ptr_->size(), X_ptr_->size()) };
    }

    num_data_points_ = X_ptr_->size();
    num_features_ = X_ptr_->front().size();
}

template <typename T>
void data_set<T>::save_data_set(const std::string &filename, const file_format_type format) const {
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

template <typename T>
void data_set<T>::scale() {
    this->scale(-1, +1);
}

template <typename T>
void data_set<T>::scale(const real_type lower, const real_type upper) {
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
 *                                   PRIVATE                                  *
 ******************************************************************************/


template <typename T>
void data_set<T>::write_libsvm_file(const std::string& filename) const {
    auto out = fmt::output_file(filename);
    // write data
    if (this->has_labels()) {
        detail::io::write_libsvm_data(out, X_ptr_, y_ptr_);
    } else {
        detail::io::write_libsvm_data(out, X_ptr_);
    }
}

template <typename T>
void data_set<T>::write_arff_file(const std::string &filename) const {
    auto out = fmt::output_file(filename);

    // write header information
    detail::io::write_arff_header(out, num_features_, this->has_labels());

    // write data
    if (this->has_labels()) {
        detail::io::write_arff_data(out, X_ptr_, y_ptr_);
    } else {
        detail::io::write_arff_data(out, X_ptr_);
    }
}

template <typename T>
void data_set<T>::read_file(const std::string &filename, file_format_type format) {
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

template <typename T>
void data_set<T>::read_libsvm_file(const std::string &filename) {
    detail::io::file_reader f{ filename, '#' };

    // parse sizes
    num_data_points_ = f.num_lines();
    num_features_ = detail::io::parse_libsvm_num_features(f, f.num_lines(), 0);

    X_ptr_ = std::make_shared<data_matrix_type>(num_data_points_, std::vector<real_type>(num_features_));
    y_ptr_ = std::make_shared<label_vector_type>(num_data_points_);

    // parse file
    const bool has_label = detail::io::read_libsvm_data(f, 0, X_ptr_, y_ptr_);

    // gamma?
    // TODO:

    // convert label values to -1/+1
    if (has_label) {
        // only if labels are present
        #pragma omp parallel for default(none) shared(y_ptr_)
        for (size_type i = 0; i < num_data_points_; ++i) {
            (*y_ptr_)[i] = operators::sign((*y_ptr_)[i]);
        }
    } else {
        y_ptr_ = nullptr;
    }
}

template <typename T>
void data_set<T>::read_arff_file(const std::string &filename) {
    detail::io::file_reader f{ filename, '%' };

    // parse arff header, structured binding
    size_type header = 0;
    size_type max_size = 0;
    bool has_label = false;
    std::tie(header, max_size, has_label) = detail::io::read_arff_header(f);

    num_data_points_ = f.num_lines() - (header + 1);
    num_features_ = has_label ? max_size - 1 : max_size;

    X_ptr_ = std::make_shared<data_matrix_type>(num_features_, std::vector<real_type>(num_features_));
    y_ptr_ = std::make_shared<label_vector_type>(num_data_points_);

    // parse file
    detail::io::read_arff_data(f, header, num_features_, max_size, has_label, X_ptr_, y_ptr_);

    // update gamma
    // TODO:

    // update shared pointer
    if (!has_label) {
        y_ptr_ = nullptr;
    }
}

// explicitly instantiate template class
template class data_set<float>;
template class data_set<double>;

}
/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the ARFF file format.
 */

#pragma once

#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader

#include "fmt/os.h"  // fmt::ostream

#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr
#include <tuple>    // std::tuple
#include <vector>   // std::vector

namespace plssvm::detail::io {

std::tuple<std::size_t, std::size_t, bool> read_arff_header(file_reader &reader);

template <typename real_type>
void read_arff_data(file_reader &reader, std::size_t header, std::size_t num_features, std::size_t max_size, bool has_label, std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr, std::shared_ptr<std::vector<real_type>> &y_ptr);

void write_arff_header(fmt::ostream &out, std::size_t num_features, bool has_labels);

template <typename real_type>
void write_arff_data(fmt::ostream &out, const std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr, const std::shared_ptr<std::vector<real_type>> &y_ptr);

template <typename real_type>
void write_arff_data(fmt::ostream &out, const std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr);

}
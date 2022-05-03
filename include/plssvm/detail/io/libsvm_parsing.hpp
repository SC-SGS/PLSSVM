/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the LIBSVM file format.
 */

#pragma once

#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::file_reader

#include "fmt/os.h"  // fmt::ostream

#include <cstddef>  // std::size_t
#include <memory>   // std::shared_ptr
#include <vector>   // std::vector

namespace plssvm::detail::io {

std::size_t parse_libsvm_num_features(file_reader &reader, std::size_t num_data_points, std::size_t start);

template <typename real_type>
bool read_libsvm_data(file_reader &reader, std::size_t start, std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr, std::shared_ptr<std::vector<real_type>> &y_ptr);

template <typename real_type>
void write_libsvm_data(fmt::ostream &out, const std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr, const std::shared_ptr<std::vector<real_type>> &y_ptr);

template <typename real_type>
void write_libsvm_data(fmt::ostream &out, const std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr);

}
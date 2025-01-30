/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Variant wrapper structs around PLSSVM classification and regression data sets. Used that we don't have to expose templates to the Python bindings.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_DATA_SET_WRAPPER_HPP_
#define PLSSVM_BINDINGS_PYTHON_DATA_SET_WRAPPER_HPP_
#pragma once

#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/data_set/regression_data_set.hpp"      // plssvm::regression_data_set

#include <cstdint>  // fixed-width integers
#include <string>   // std::string
#include <utility>  // std::move
#include <variant>  // std::variant
#include <vector>   // std::vector

namespace plssvm::bindings::python::util {

/**
 * @brief A wrapper struct encapsulating all possible classification data sets.
 */
struct classification_data_set_wrapper {
    /// A std::variant containing all possible classification data set label types.
    using possible_vector_types = std::variant<std::vector<bool>,           // np.bool
                                               std::vector<std::int8_t>,    // np.int8
                                               std::vector<std::uint8_t>,   // np.uint8
                                               std::vector<std::int16_t>,   // np.int16
                                               std::vector<std::uint16_t>,  // np.uint16
                                               std::vector<std::int32_t>,   // np.int32
                                               std::vector<std::uint32_t>,  // np.uint32
                                               std::vector<std::int64_t>,   // np.int64
                                               std::vector<std::uint64_t>,  // np.uint64
                                               std::vector<float>,          // np.float32
                                               std::vector<double>,         // np.float64
                                               std::vector<std::string>>;   // np.string

    /// A std::variant containing all possible classification data set types.
    using possible_data_set_types = std::variant<plssvm::classification_data_set<bool>,           // np.bool
                                                 plssvm::classification_data_set<std::int8_t>,    // np.int8
                                                 plssvm::classification_data_set<std::uint8_t>,   // np.uint8
                                                 plssvm::classification_data_set<std::int16_t>,   // np.int16
                                                 plssvm::classification_data_set<std::uint16_t>,  // np.uint16
                                                 plssvm::classification_data_set<std::int32_t>,   // np.int32
                                                 plssvm::classification_data_set<std::uint32_t>,  // np.uint32
                                                 plssvm::classification_data_set<std::int64_t>,   // np.int64
                                                 plssvm::classification_data_set<std::uint64_t>,  // np.uint64
                                                 plssvm::classification_data_set<float>,          // np.float32
                                                 plssvm::classification_data_set<double>,         // np.float64
                                                 plssvm::classification_data_set<std::string>>;   // np.str

    /**
     * @brief Construct a new classification data set by setting the active std::variant member.
     * @tparam T the label type of the classification data set
     * @param[in] d the classification data set
     */
    template <typename T>
    classification_data_set_wrapper(plssvm::classification_data_set<T> d) :
        data_set{ std::move(d) } { }

    /**
     * @brief Construct a new classification data set using the provided std::variant.
     * @param[in] d the classification data set variant
     */
    classification_data_set_wrapper(possible_data_set_types d) :
        data_set{ std::move(d) } { }

    /// The actual classification data set (active type in the std::variant).
    possible_data_set_types data_set;
};

/**
 * @brief A wrapper struct encapsulating all possible regression data sets.
 */
struct regression_data_set_wrapper {
    /// A std::variant containing all possible regression data set label types.
    using possible_vector_types = std::variant<std::vector<std::int16_t>,   // np.int16
                                               std::vector<std::uint16_t>,  // np.uint16
                                               std::vector<std::int32_t>,   // np.int32
                                               std::vector<std::uint32_t>,  // np.uint32
                                               std::vector<std::int64_t>,   // np.int64
                                               std::vector<std::uint64_t>,  // np.uint64
                                               std::vector<float>,          // np.float32
                                               std::vector<double>>;        // np.float64

    /// A std::variant containing all possible regression data set types.
    using possible_data_set_types = std::variant<plssvm::regression_data_set<std::int16_t>,   // np.int16
                                                 plssvm::regression_data_set<std::uint16_t>,  // np.uint16
                                                 plssvm::regression_data_set<std::int32_t>,   // np.int32
                                                 plssvm::regression_data_set<std::uint32_t>,  // np.uint32
                                                 plssvm::regression_data_set<std::int64_t>,   // np.int64
                                                 plssvm::regression_data_set<std::uint64_t>,  // np.uint64
                                                 plssvm::regression_data_set<float>,          // np.float32
                                                 plssvm::regression_data_set<double>>;        // np.float64

    /**
     * @brief Construct a new regression data set by setting the active std::variant member.
     * @tparam T the label type of the regression data set
     * @param[in] d the regression data set
     */
    template <typename T>
    regression_data_set_wrapper(plssvm::regression_data_set<T> d) :
        data_set{ std::move(d) } { }

    /**
     * @brief Construct a new regression data set using the provided std::variant.
     * @param[in] d the regression data set variant
     */
    regression_data_set_wrapper(possible_data_set_types d) :
        data_set{ std::move(d) } { }

    /// The actual regression data set (active type in the std::variant).
    possible_data_set_types data_set;
};

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_DATA_SET_WRAPPER_HPP_

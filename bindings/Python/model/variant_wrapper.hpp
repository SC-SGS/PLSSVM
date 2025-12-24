/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Variant wrapper structs around PLSSVM classification and regression models. Used that we don't have to expose templates to the Python bindings.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_MODEL_VARIANT_WRAPPER_HPP_
#define PLSSVM_BINDINGS_PYTHON_MODEL_VARIANT_WRAPPER_HPP_
#pragma once

#include "plssvm/model/classification_model.hpp"  // plssvm::classification_model
#include "plssvm/model/regression_model.hpp"      // plssvm::regression_model

#include <cstdint>  // fixed-width integers
#include <string>   // std::string
#include <utility>  // std::move
#include <variant>  // std::variant

namespace plssvm::bindings::python::util {

/**
 * @brief A wrapper struct encapsulating all possible classification models.
 */
struct classification_model_wrapper {
    /// A std::variant containing all possible classification model types.
    using possible_model_types = std::variant<plssvm::classification_model<bool>,           // np.bool
                                              plssvm::classification_model<std::int8_t>,    // np.int8
                                              plssvm::classification_model<std::uint8_t>,   // np.uint8
                                              plssvm::classification_model<std::int16_t>,   // np.int16
                                              plssvm::classification_model<std::uint16_t>,  // np.uint16
                                              plssvm::classification_model<std::int32_t>,   // np.int32
                                              plssvm::classification_model<std::uint32_t>,  // np.uint32
                                              plssvm::classification_model<std::int64_t>,   // np.int64
                                              plssvm::classification_model<std::uint64_t>,  // np.uint64
                                              plssvm::classification_model<float>,          // np.float32
                                              plssvm::classification_model<double>,         // np.float64
                                              plssvm::classification_model<std::string>>;   // np.str

    /**
     * @brief Construct a new classification model by setting the active std::variant member.
     * @tparam T the label type of the classification model
     * @param[in] m the classification model
     */
    template <typename T>
    explicit classification_model_wrapper(plssvm::classification_model<T> m) :
        model{ std::move(m) } { }

    /**
     * @brief Construct a new classification model using the provided std::variant.
     * @param[in] m the classification model variant
     */
    explicit classification_model_wrapper(possible_model_types m) :
        model{ std::move(m) } { }

    /// The actual classification model (active type in the std::variant).
    possible_model_types model;
};

/**
 * @brief A wrapper struct encapsulating all possible regression models.
 */
struct regression_model_wrapper {
    /// A std::variant containing all possible regression model types.
    using possible_model_types = std::variant<plssvm::regression_model<std::int16_t>,  // np.int16
                                              plssvm::regression_model<std::int32_t>,  // np.int32
                                              plssvm::regression_model<std::int64_t>,  // np.int64
                                              plssvm::regression_model<float>,         // np.float32
                                              plssvm::regression_model<double>>;       // np.float64

    /**
     * @brief Construct a new regression model by setting the active std::variant member.
     * @tparam T the label type of the regression model
     * @param[in] m the regression model
     */
    template <typename T>
    explicit regression_model_wrapper(plssvm::regression_model<T> m) :
        model{ std::move(m) } { }

    /**
     * @brief Construct a new regression model using the provided std::variant.
     * @param[in] m the regression model variant
     */
    explicit regression_model_wrapper(possible_model_types m) :
        model{ std::move(m) } { }

    /// The actual regression model (active type in the std::variant).
    possible_model_types model;
};

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_MODEL_VARIANT_WRAPPER_HPP_

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them.
 */

#pragma once

#include "plssvm/constants.hpp"       // plssvm::verbose
#include "plssvm/data_set.hpp"        // plssvm::data_set
#include "plssvm/detail/utility.hpp"  // plssvm::detail::to_underlying, plssvm::detail::remove_cvref_t
#include "plssvm/kernel_types.hpp"    // plssvm::kernel_type
#include "plssvm/model.hpp"           // plssvm::model
#include "plssvm/parameter.hpp"       // plssvm::parameter

#include "fmt/core.h"     // fmt::format, fmt::print
#include "igor/igor.hpp"  // igor::parser

#include <chrono>       // std::chrono::{steady_clock, duration_cast}
#include <cstddef>      // std::size_t
#include <cstdio>       // stderr
#include <iostream>     // std::clog, std::endl
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v, std::is_convertible_v
#include <utility>      // std::move, std::forward
#include <vector>       // std::vector

namespace plssvm {

// TODO: remove T ??!??

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the floating point type of the data
 */
template <typename T>
class csvm {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
    /// The unsigned size type.
    using size_type = std::size_t;

    /**
     * @brief Construct a C-SVM using the SVM parameter @p params.
     * @details Uses the default SVM parameter if none are provided.
     * @param[in] params the SVM parameter
     */
    explicit csvm(parameter<real_type> params = {});
    /**
     * @brief Construct a C-SVM using named-parameters with the @p kernel type.
     * @tparam Args the type of the named-parameters
     * @param[in] kernel the kernel type to use
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args>
    explicit csvm(kernel_type kernel, Args &&...named_args);

    /**
     * @brief Default copy-constructor since a virtual destructor has been declared.
     */
    csvm(const csvm &) = default;
    /**
     * @brief Default move-constructor since a virtual destructor has been declared.
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~csvm() = default;

    /**
     * @brief Return the currently used SVM parameter.
     * @return the SVM parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] parameter<real_type> get_params() const noexcept { return params_; }
    // TODO: set_params?

    ////////////////////////////////////////////////////////////////////////////////
    ////                               fit model                                ////
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Fit a model using the current SVM on the @p data.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @tparam Args the type of the potential additional parameters
     * @param[in] data the data used to train the SVM model
     * @param[in] named_args the potential additional parameters (`epsilon` and/or `max_iter`)
     * @return the learned model (`[[nodiscard]]`)
     */
    template <typename label_type, typename... Args>
    [[nodiscard]] model<real_type, label_type> fit(const data_set<real_type, label_type> &data, Args &&...named_args) const;

    ////////////////////////////////////////////////////////////////////////////////
    ////                           predict and score                            ////
    ////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Predict the labels for the @p data set using the @p model.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learnt model
     * @param[in] data the data to predict the labels for
     * @return the predicted labels (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data);

    /**
     * @brief Calculate the accuracy of the @p model.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learnt model
     * @return the accuracy of the model (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model);
    /**
     * @brief Calculate the accuracy of the labeled @p data set using the @p model.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learnt model
     * @param data the labeled data set to score
     * @return the accuracy of the labeled @p data (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model, const data_set<real_type> &data);

  protected:
    /////////////////////////////////////////////////////////////////////////////////////////////////
    ////  pure virtual functions, must be implemented for all subclasses; doing the actual work  ////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Solves using a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] params the SVM parameters used in the respective kernel functions
     * @param[in] A the matrix of the equation \f$Ax = b\f$ (symmetric positive definite)
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] eps error tolerance
     * @param[in] max_iter the maximum number of CG iterations
     * @return a pair of [the result vector x, the resulting bias] (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::pair<std::vector<real_type>, real_type> solve_system_of_linear_equations(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, real_type eps, size_type max_iter) const = 0;
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] params the SVM parameters used in the respective kernel functions
     * @param[in] support_vectors the previously learned support vectors
     * @param[in] alpha the alpha values (weights) associated with the support vectors
     * @param[in] rho the rho value determined after training the model
     * @param[in,out] w the normal vector to speedup prediction in case of the linear kernel function, an empty vector in case of the polynomial or rbf kernel
     * @param[in] predict_points the points to predict
     * @return a vector filled with the predictions (not the actual labels!) (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<real_type> predict_values(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const = 0;

  private:
    /**
     * @brief Perform some sanity checks on the passed SVM parameters.
     */
    void sanity_check_parameter();

    template <typename ExpectedType, typename IgorParser, typename ProvidedType>
    ExpectedType get_value_from_named_parameter(const IgorParser &parser, const ProvidedType &named_arg);

    /// The SVM parameter (e.g., cost, degree, gamma, coef0) currently in use.
    parameter<real_type> params_{};
};

template <typename T>
csvm<T>::csvm(parameter<real_type> params) :
    params_{ std::move(params) } {
    this->sanity_check_parameter();
}

template <typename T>
template <typename... Args>
csvm<T>::csvm(kernel_type kernel, Args &&...named_args) {
    igor::parser parser{ std::forward<Args>(named_args)... };

    // set kernel type
    params_.kernel = kernel;

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(gamma, degree, coef0, cost, sycl_implementation_type, sycl_kernel_invocation_type), "An illegal named parameter has been passed!");

    // shorthand function for emitting a warning if a provided parameter is not used by the current kernel function
    [[maybe_unused]] const auto print_warning = [kernel](const std::string_view param_name) {
        std::clog << fmt::format("{} parameter provided, which is not used in the {} kernel ({})!", param_name, kernel, kernel_type_to_math_string(kernel)) << std::endl;
    };

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(gamma)) {
        // get the value of the provided named parameter
        params_.gamma = get_value_from_named_parameter<typename decltype(params_.gamma)::value_type>(parser, gamma);
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear) {
            print_warning("gamma");
        }
    }
    if constexpr (parser.has(degree)) {
        // get the value of the provided named parameter
        params_.degree = get_value_from_named_parameter<typename decltype(params_.degree)::value_type>(parser, degree);
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear || kernel == kernel_type::rbf) {
            print_warning("degree");
        }
    }
    if constexpr (parser.has(coef0)) {
        // get the value of the provided named parameter
        params_.coef0 = get_value_from_named_parameter<typename decltype(params_.coef0)::value_type>(parser, coef0);
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear || kernel == kernel_type::rbf) {
            print_warning("coef0");
        }
    }
    if constexpr (parser.has(cost)) {
        // get the value of the provided named parameter
        params_.cost = get_value_from_named_parameter<typename decltype(params_.cost)::value_type>(parser, cost);
    }

    // check if parameters make sense
    this->sanity_check_parameter();
}

template <typename T>
template <typename label_type, typename... Args>
auto csvm<T>::fit(const data_set<real_type, label_type> &data, Args &&...named_args) const -> model<real_type, label_type> {
    igor::parser parser{ std::forward<Args>(named_args)... };

    // set default values
    default_value epsilon_val{ default_init<real_type>{ 0.001 } };
    default_value max_iter_val{ default_init<size_type>{ data.num_data_points() } };

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(epsilon, max_iter), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(epsilon)) {
        // get the value of the provided named parameter
        epsilon_val = get_value_from_named_parameter<typename decltype(epsilon_val)::value_type>(parser, epsilon);
        // check if value makes sense
        if (epsilon_val <= real_type{ 0.0 }) {
            throw invalid_parameter_exception{ fmt::format("epsilon must be less than 0.0, but is {}!", epsilon_val) };
        }
    }
    if constexpr (parser.has(max_iter)) {
        // get the value of the provided named parameter
        max_iter_val = get_value_from_named_parameter<typename decltype(max_iter_val)::value_type>(parser, max_iter);
        // check if value makes sense
        if (max_iter_val == size_type{ 0 }) {
            throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", max_iter_val) };
        }
    }

    // start fitting the data set using a C-SVM

    if (!data.has_labels()) {
        throw exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
    }

    // copy parameter and set gamma if necessary
    parameter<real_type> params{ static_cast<parameter<real_type>>(params_) };
    if (params.gamma.is_default()) {
        // no gamma provided -> use default value which depends on the number of features of the data set
        params.gamma = real_type{ 1.0 } / data.num_features();
    }

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // create model
    model<real_type, label_type> csvm_model{ params, data };

    // solve the minimization problem
    std::tie(*csvm_model.alpha_ptr_, csvm_model.rho_) = solve_system_of_linear_equations(params, data.data(), *data.y_ptr_, epsilon_val.value(), max_iter_val.value());

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << fmt::format("Solved minimization problem (r = b - Ax) using the Conjugate Gradient (CG) methode in {}.", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)) << std::endl;
    }

    return csvm_model;
}

template <typename T>
template <typename label_type>
auto csvm<T>::predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) -> std::vector<label_type> {
    if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // predict values
    const std::vector<real_type> predicted_values = predict_values(model.params_, model.data_.data(), *model.alpha_ptr_, model.rho_, *model.w_, data.data());

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(predicted_values.size());

#pragma omp parallel for default(none) shared(predicted_labels, predicted_values, model) if (!std::is_same_v <label_type, bool>)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        predicted_labels[i] = model.data_.mapping_->get_label_by_mapped_value(plssvm::operators::sign(predicted_values[i]));
    }

    return predicted_labels;
}

template <typename T>
template <typename label_type>
auto csvm<T>::score(const model<real_type, label_type> &model) -> real_type {
    return this->score(model, model.data_);
}

template <typename T>
template <typename label_type>
auto csvm<T>::score(const model<real_type, label_type> &model, const data_set<real_type> &data) -> real_type {
    if (!data.has_labels()) {
        throw exception{ "the data set to score must have labels set!" };
    } else if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // predict labels
    const std::vector<label_type> predicted_labels = predict(model, data);
    // correct labels
    const std::vector<label_type> &correct_labels = model.labels().value();

    // calculate the accuracy
    size_type correct{ 0 };
#pragma omp parallel for reduction(+ \
                                   : correct) default(none) shared(predicted_labels, correct_labels)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == correct_labels[i]) {
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(predicted_labels.size());
}

template <typename T>
void csvm<T>::sanity_check_parameter() {
    // kernel: valid kernel function
    if (params_.kernel != kernel_type::linear && params_.kernel != kernel_type::polynomial && params_.kernel != kernel_type::rbf) {
        throw invalid_parameter_exception{ fmt::format("Invalid kernel function {} given!", detail::to_underlying(params_.kernel)) };
    }

    // gamma: must be greater than 0 IF explicitly provided, but only in the polynomial and rbf kernel
    if ((params_.kernel == kernel_type::polynomial || params_.kernel == kernel_type::rbf) && !params_.gamma.is_default() && params_.gamma.value() <= real_type{ 0.0 }) {
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0.0, but is {}!", params_.gamma) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

template <typename T> template <typename ExpectedType, typename IgorParser, typename NamedArgType>
ExpectedType csvm<T>::get_value_from_named_parameter(const IgorParser &parser, const NamedArgType &named_arg) {
    using parsed_named_arg_type = detail::remove_cvref_t<decltype(parser(named_arg))>;
    // check whether a plssvm::default_value (e.g., plssvm::default_value<double>) or unwrapped normal value (e.g., double) has been provided
    if constexpr (is_default_value_v<parsed_named_arg_type>) {
        static_assert(std::is_convertible_v<typename parsed_named_arg_type::value_type, ExpectedType>, "Cannot convert the wrapped default value to the expected type!");
        // a plssvm::default_value has been provided (e.g., plssvm::default_value<double>)
        // if the provided plssvm::default_value only contains the default value, do nothing
        if (!parser(named_arg).is_default()) {
            return static_cast<ExpectedType>(parser(named_arg).value());
        }
    } else if constexpr (std::is_convertible_v<parsed_named_arg_type, ExpectedType>) {
        // an unwrapped value has been provided (e.g., double)
        return static_cast<ExpectedType>(parser(named_arg));
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "The named parameter must be of type plssvm::default_value or a built-in type!");
    }
    // may never been reached
    return ExpectedType{};
}

}  // namespace plssvm

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

#ifndef PLSSVM_CSVM_HPP_
#define PLSSVM_CSVM_HPP_
#pragma once

#include "plssvm/backend_types.hpp"          // plssvm::backend_type
#include "plssvm/constants.hpp"              // plssvm::verbose
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/default_value.hpp"          // plssvm::default_value, plssvm::default_init, plssvm::is_default_value_v
#include "plssvm/detail/operators.hpp"       // plssvm::operators::sign
#include "plssvm/detail/utility.hpp"         // plssvm::detail::{to_underlying, remove_cvref_t, always_false_v, unreachable}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type, plssvm::kernel_function_type_to_math_string
#include "plssvm/model.hpp"                  // plssvm::model
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"     // fmt::format, fmt::print
#include "igor/igor.hpp"  // igor::parser

#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast}
#include <cstddef>      // std::size_t
#include <iostream>     // std::clog, std::endl
#include <string_view>  // std::string_view
#include <tuple>        // std::tie
#include <type_traits>  // std::is_same_v, std::is_convertible_v, std::false_type
#include <utility>      // std::move, std::forward
#include <vector>       // std::vector

namespace plssvm {

// TODO: exception docu

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the floating point type of the data
 */
class csvm {
  public:
    /**
     * @brief Construct a C-SVM using the SVM parameter @p params.
     * @details Uses the default SVM parameter if none are provided.
     * @param[in] params the SVM parameter
     */
    explicit csvm(parameter params = {});
    /**
     * @brief Construct a C-SVM using named-parameters with the @p kernel type.
     * @tparam Args the type of the named-parameters
     * @param[in] kernel the kernel type to use
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args>
    explicit csvm(kernel_function_type kernel, Args &&...named_args);

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
    [[nodiscard]] parameter get_params() const noexcept { return params_; } // static_assert

    /**
     * @brief Override the old SVM parameter with the new @p params.
     * @param[in] params the new SVM parameter to use
     */
    void set_params(parameter params) noexcept { params_ = params; }
    /**
     * @brief Override the old SVM parameter with the new ones given as named parameters in @ named_args.
     * @tparam Args the type of the named-parameters
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args>
    void set_params(Args &&...named_args);

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
    template <typename real_type, typename label_type, typename... Args>
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
    template <typename real_type, typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const;

    /**
     * @brief Calculate the accuracy of the @p model.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learnt model
     * @return the accuracy of the model (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model) const;
    /**
     * @brief Calculate the accuracy of the labeled @p data set using the @p model.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learnt model
     * @param data the labeled data set to score
     * @return the accuracy of the labeled @p data (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const;

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
    [[nodiscard]] virtual std::pair<std::vector<float>, float> solve_system_of_linear_equations(const detail::parameter<float> &params, const std::vector<std::vector<float>> &A, std::vector<float> b, float eps, unsigned long long max_iter) const = 0;
    /**
     * @copydoc plssvm::csvm::solve_system_of_linear_equations
     */
    [[nodiscard]] virtual std::pair<std::vector<double>, double> solve_system_of_linear_equations(const detail::parameter<double> &params, const std::vector<std::vector<double>> &A, std::vector<double> b, double eps, unsigned long long max_iter) const = 0;
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
    [[nodiscard]] virtual std::vector<float> predict_values(const detail::parameter<float> &params, const std::vector<std::vector<float>> &support_vectors, const std::vector<float> &alpha, float rho, std::vector<float> &w, const std::vector<std::vector<float>> &predict_points) const = 0;
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] virtual std::vector<double> predict_values(const detail::parameter<double> &params, const std::vector<std::vector<double>> &support_vectors, const std::vector<double> &alpha, double rho, std::vector<double> &w, const std::vector<std::vector<double>> &predict_points) const = 0;

  private:
    /**
     * @brief Perform some sanity checks on the passed SVM parameters.
     */
    void sanity_check_parameter() const;

    /**
     * @brief Parse the value hold be @p named_arg and return it converted to the @p ExpectedType.
     * @tparam ExpectedType the type the value of the named argument should be converted to
     * @tparam IgorParser the type of the named argument parser
     * @tparam ProvidedType the type of the named argument (necessary since their are struct tags)
     * @param[in] parser the named argument parser
     * @param[in] named_arg the named argument
     * @return the value of @p named_arg converted to @p ExpectedType (`[[nodiscard]]`)
     */
    template <typename ExpectedType, typename IgorParser, typename ProvidedType>
    [[nodiscard]] ExpectedType get_value_from_named_parameter(const IgorParser &parser, const ProvidedType &named_arg) const;

    /// The SVM parameter (e.g., cost, degree, gamma, coef0) currently in use.
    parameter params_{};
};

inline csvm::csvm(parameter params) :
    params_{ params } {
    this->sanity_check_parameter();
}

template <typename... Args>
csvm::csvm(kernel_function_type kernel, Args &&...named_args) {
    this->set_params(kernel_type = kernel, std::forward<Args>(named_args)...);
}

template <typename... Args>
void csvm::set_params(Args &&...named_args) {
    igor::parser parser{ std::forward<Args>(named_args)... };

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed // TODO: SYCL
    static_assert(!parser.has_other_than(kernel_type, gamma, degree, coef0, cost, sycl_implementation_type, sycl_kernel_invocation_type), "An illegal named parameter has been passed!");

    // shorthand function for emitting a warning if a provided parameter is not used by the current kernel function
    [[maybe_unused]] const auto print_warning = [](const std::string_view param_name, const kernel_function_type kernel) {
        std::clog << fmt::format("{} parameter provided, which is not used in the {} kernel ({})!", param_name, kernel, kernel_function_type_to_math_string(kernel)) << std::endl;
    };

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(kernel_type)) {
        // get the value of the provided named parameter
        params_.kernel_type = get_value_from_named_parameter<typename decltype(params_.kernel_type)::value_type>(parser, kernel_type);
    }
    if constexpr (parser.has(gamma)) {
        // get the value of the provided named parameter
        params_.gamma = get_value_from_named_parameter<typename decltype(params_.gamma)::value_type>(parser, gamma);
        // runtime check: the value may only be used with a specific kernel type
        if (params_.kernel_type == kernel_function_type::linear) {
            print_warning("gamma", params_.kernel_type);
        }
    }
    if constexpr (parser.has(degree)) {
        // get the value of the provided named parameter
        params_.degree = get_value_from_named_parameter<typename decltype(params_.degree)::value_type>(parser, degree);
        // runtime check: the value may only be used with a specific kernel type
        if (params_.kernel_type == kernel_function_type::linear || params_.kernel_type == kernel_function_type::rbf) {
            print_warning("degree", params_.kernel_type);
        }
    }
    if constexpr (parser.has(coef0)) {
        // get the value of the provided named parameter
        params_.coef0 = get_value_from_named_parameter<typename decltype(params_.coef0)::value_type>(parser, coef0);
        // runtime check: the value may only be used with a specific kernel type
        if (params_.kernel_type == kernel_function_type::linear || params_.kernel_type == kernel_function_type::rbf) {
            print_warning("coef0", params_.kernel_type);
        }
    }
    if constexpr (parser.has(cost)) {
        // get the value of the provided named parameter
        params_.cost = get_value_from_named_parameter<typename decltype(params_.cost)::value_type>(parser, cost);
    }

    // check if parameters make sense
    this->sanity_check_parameter();
}

template <typename real_type, typename label_type, typename... Args>
model<real_type, label_type> csvm::fit(const data_set<real_type, label_type> &data, Args &&...named_args) const {
    igor::parser parser{ std::forward<Args>(named_args)... };

    // set default values
    default_value epsilon_val{ default_init<real_type>{ 0.001 } };
    default_value max_iter_val{ default_init<unsigned long long>{ data.num_data_points() } };

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
        if (epsilon_val <= static_cast<typename decltype(epsilon_val)::value_type>(0)) {
            throw invalid_parameter_exception{ fmt::format("epsilon must be less than 0.0, but is {}!", epsilon_val) };
        }
    }
    if constexpr (parser.has(max_iter)) {
        // get the value of the provided named parameter
        max_iter_val = get_value_from_named_parameter<typename decltype(max_iter_val)::value_type>(parser, max_iter);
        // check if value makes sense
        if (max_iter_val == static_cast<typename decltype(max_iter_val)::value_type>(0)) {
            throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", max_iter_val) };
        }
    }

    // start fitting the data set using a C-SVM

    if (!data.has_labels()) {
        throw invalid_parameter_exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
    }

    // copy parameter and set gamma if necessary
    parameter params{ params_ };
    if (params.gamma.is_default()) {
        // no gamma provided -> use default value which depends on the number of features of the data set
        params.gamma = 1.0 / data.num_features();
    }

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // create model
    model<real_type, label_type> csvm_model{ params, data };

    // solve the minimization problem
    std::tie(*csvm_model.alpha_ptr_, csvm_model.rho_) = solve_system_of_linear_equations(static_cast<detail::parameter<real_type>>(params), data.data(), *data.y_ptr_, epsilon_val.value(), max_iter_val.value());

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << fmt::format("Solved minimization problem (r = b - Ax) using the Conjugate Gradient (CG) methode in {}.", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)) << std::endl;
    }

    return csvm_model;
}

template <typename real_type, typename label_type>
std::vector<label_type> csvm::predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const {
    if (model.num_features() != data.num_features()) {
        throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
    }

    // predict values
    const std::vector<real_type> predicted_values = predict_values(static_cast<detail::parameter<real_type>>(model.params_), model.data_.data(), *model.alpha_ptr_, model.rho_, *model.w_, data.data());

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(predicted_values.size());

    #pragma omp parallel for default(none) shared(predicted_labels, predicted_values, model) if (!std::is_same_v<label_type, bool>)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        predicted_labels[i] = model.data_.mapping_->get_label_by_mapped_value(plssvm::operators::sign(predicted_values[i]));
    }

    return predicted_labels;
}

template <typename real_type, typename label_type>
real_type csvm::score(const model<real_type, label_type> &model) const {
    return this->score(model, model.data_);
}

template <typename real_type, typename label_type>
real_type csvm::score(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const {
    // the data set must contain labels in order to score the learned model
    if (!data.has_labels()) {
        throw invalid_parameter_exception{ "The data set to score must have labels!" };
    }
    // the number of features must be equal
    if (model.num_features() != data.num_features()) {
        throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
    }

    // predict labels
    const std::vector<label_type> predicted_labels = predict(model, data);
    // correct labels
    const std::vector<label_type> &correct_labels = data.labels().value();

    // calculate the accuracy
    typename std::vector<label_type>::size_type correct{ 0 };
    #pragma omp parallel for reduction(+ : correct) default(none) shared(predicted_labels, correct_labels)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == correct_labels[i]) {
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(predicted_labels.size());
}

inline void csvm::sanity_check_parameter() const {
    // kernel: valid kernel function
    if (params_.kernel_type != kernel_function_type::linear && params_.kernel_type != kernel_function_type::polynomial && params_.kernel_type != kernel_function_type::rbf) {
        throw invalid_parameter_exception{ fmt::format("Invalid kernel function {} given!", detail::to_underlying(params_.kernel_type)) };
    }

    // gamma: must be greater than 0 IF explicitly provided, but only in the polynomial and rbf kernel
    if ((params_.kernel_type == kernel_function_type::polynomial || params_.kernel_type == kernel_function_type::rbf) &&
        !params_.gamma.is_default() && params_.gamma.value() <= 0.0) {
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0.0, but is {}!", params_.gamma) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

template <typename ExpectedType, typename IgorParser, typename NamedArgType>
ExpectedType csvm::get_value_from_named_parameter(const IgorParser &parser, const NamedArgType &named_arg) const {
    using parsed_named_arg_type = detail::remove_cvref_t<decltype(parser(named_arg))>;
    // check whether a plssvm::default_value (e.g., plssvm::default_value<double>) or unwrapped normal value (e.g., double) has been provided
    if constexpr (is_default_value_v<parsed_named_arg_type>) {
        static_assert(std::is_convertible_v<typename parsed_named_arg_type::value_type, ExpectedType>, "Cannot convert the wrapped default value to the expected type!");
        // a plssvm::default_value has been provided (e.g., plssvm::default_value<double>)
        return static_cast<ExpectedType>(parser(named_arg).value());
    } else if constexpr (std::is_convertible_v<parsed_named_arg_type, ExpectedType>) {
        // an unwrapped value has been provided (e.g., double)
        return static_cast<ExpectedType>(parser(named_arg));
    } else {
        static_assert(plssvm::detail::always_false_v<ExpectedType>, "The named parameter must be of type plssvm::default_value or a built-in type!");
    }
    // may never been reached
    detail::unreachable();
}

namespace detail {

/**
 * @brief Sets the `value` to `false` since the given type @p T is either not a C-SVM or the C-SVM using the requested backend isn't available.
 * @tparam T the type of the C-SVM
 */
template <typename T>
struct csvm_backend_exists : std::false_type {};

}

/**
 * @brief Sets the value of the `value` member to `true` if @p T is a C-SVM using an available backend. Ignores any const, volatile, and reference qualifiers.
 * @tparam T the type of the C-SVM
 */
template <typename T>
struct csvm_backend_exists : detail::csvm_backend_exists<detail::remove_cvref_t<T>> {};

/**
 * @brief Sets the value of the `value` member to `true` if @p T is a C-SVM using an available backend. Ignores any const, volatile, and reference qualifiers.
 * @tparam T the type of the C-SVM
 */
template <typename T>
constexpr bool csvm_backend_exists_v = csvm_backend_exists<T>::value;



}  // namespace plssvm

#endif  // PLSSVM_CSVM_HPP_
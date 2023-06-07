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

#include "plssvm/classification_types.hpp"        // plssvm::classification_type, plssvm::classification_type_to_full_string
#include "plssvm/data_set.hpp"                    // plssvm::data_set
#include "plssvm/default_value.hpp"               // plssvm::default_value, plssvm::default_init
#include "plssvm/detail/logger.hpp"               // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/operators.hpp"            // plssvm::operators::sign
#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::performance_tracker
#include "plssvm/detail/type_traits.hpp"          // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"              // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/model.hpp"                       // plssvm::model
#include "plssvm/parameter.hpp"                   // plssvm::parameter, plssvm::detail::{get_value_from_named_parameter, has_only_parameter_named_args_v}
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "fmt/core.h"                             // fmt::format
#include "igor/igor.hpp"                          // igor::parser

#include <algorithm>                              // std::max_element
#include <chrono>                                 // std::chrono::{time_point, steady_clock, duration_cast}
#include <iostream>                               // std::cout, std::endl
#include <iterator>                               // std::distance
#include <tuple>                                  // std::tie
#include <type_traits>                            // std::enable_if_t, std::is_same_v, std::is_convertible_v, std::false_type
#include <utility>                                // std::pair, std::forward
#include <vector>                                 // std::vector

namespace plssvm {

/**
 * @example csvm_examples.cpp
 * @brief A few examples regarding the plssvm::csvm class.
 */

/**
 * @brief Base class for all C-SVM backends.
 * @details This class implements all features shared between all C-SVM backends. It defines the whole public API of a C-SVM.
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
     * @brief Construct a C-SVM forwarding all parameters @p args to the plssvm::parameter constructor.
     * @tparam Args the type of the (named-)parameters
     * @param[in] args the parameters used to construct a plssvm::parameter
     */
    template <typename... Args>
    explicit csvm(Args &&...args);

    /**
     * @brief Delete copy-constructor since a CSVM is a move-only type.
     */
    csvm(const csvm &) = delete;
    /**
     * @brief Default move-constructor since a virtual destructor has been declared.
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @brief Delete copy-assignment operator since a CSVM is a move-only type.
     * @return `*this`
     */
    csvm &operator=(const csvm &) = delete;
    /**
     * @brief Default move-assignment operator since a virtual destructor has been declared.
     * @return `*this`
     */
    csvm &operator=(csvm &&) noexcept = default;
    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~csvm() = default;

    /**
     * @brief Return the target platform (i.e, CPU or GPU including the vendor) this SVM runs on.
     * @return the target platform (`[[nodiscard]]`)
     */
    [[nodiscard]] target_platform get_target_platform() const noexcept { return target_;}
    /**
     * @brief Return the currently used SVM parameter.
     * @return the SVM parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] parameter get_params() const noexcept { return params_; }

    /**
     * @brief Override the old SVM parameter with the new plssvm::parameter @p params.
     * @param[in] params the new SVM parameter to use
     */
    void set_params(parameter params) noexcept { params_ = params; }
    /**
     * @brief Override the old SVM parameter with the new ones given as named parameters in @p named_args.
     * @tparam Args the type of the named-parameters
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_parameter_named_args_v<Args...>)>
    void set_params(Args &&...named_args);

    //*************************************************************************************************************************************//
    //                                                              fit model                                                              //
    //*************************************************************************************************************************************//
    /**
     * @brief Fit a model using the current SVM on the @p data.
     * @tparam real_type the type of the data (`float` or `double`)
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @tparam Args the type of the potential additional parameters
     * @param[in] data the data used to train the SVM model
     * @param[in] named_args the potential additional parameters (`epsilon`, `max_iter`, and `classification`)
     * @throws plssvm::invalid_parameter_exception if the provided value for `epsilon` is greater or equal than zero
     * @throws plssvm::invlaid_parameter_exception if the provided maximum number of iterations is less or equal than zero
     * @throws plssvm::invalid_parameter_exception if the training @p data does **not** include labels
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::solve_system_of_linear_equations`
     * @return the learned model (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type, typename... Args>
    [[nodiscard]] model<real_type, label_type> fit(const data_set<real_type, label_type> &data, Args &&...named_args) const;

    //*************************************************************************************************************************************//
    //                                                          predict and score                                                          //
    //*************************************************************************************************************************************//
    /**
     * @brief Predict the labels for the @p data set using the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam real_type the type of the data (`float` or `double`)
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @param[in] data the data to predict the labels for
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the predicted labels (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const;

    /**
     * @brief Calculate the accuracy of the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam real_type the type of the data (`float` or `double`)
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the accuracy of the model (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model) const;
    /**
     * @brief Calculate the accuracy of the labeled @p data set using the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam real_type the type of the data (`float` or `double`)
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @param[in] data the labeled data set to score
     * @throws plssvm::invalid_parameter_exception if the @p data to score has no labels
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the accuracy of the labeled @p data (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const;

  protected:
    //*************************************************************************************************************************************//
    //                        pure virtual functions, must be implemented for all subclasses; doing the actual work                        //
    //*************************************************************************************************************************************//
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Uses a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] params the SVM parameters used in the respective kernel functions
     * @param[in] A the matrix of the equation \f$Ax = b\f$ (symmetric positive definite)
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] eps the error tolerance
     * @param[in] max_iter the maximum number of CG iterations
     * @throws plssvm::exception any exception thrown by the backend's implementation
     * @return a pair of [the result vector x, the resulting bias] (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::pair<std::vector<std::vector<float>>, std::vector<float>> solve_system_of_linear_equations(const detail::parameter<float> &params, const std::vector<std::vector<float>> &A, std::vector<std::vector<float>> B, float eps, unsigned long long max_iter) const = 0;
    /**
     * @copydoc plssvm::csvm::solve_system_of_linear_equations
     */
    [[nodiscard]] virtual std::pair<std::vector<std::vector<double>>, std::vector<double>> solve_system_of_linear_equations(const detail::parameter<double> &params, const std::vector<std::vector<double>> &A, std::vector<std::vector<double>> B, double eps, unsigned long long max_iter) const = 0;
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @param[in] params the SVM parameters used in the respective kernel functions
     * @param[in] support_vectors the previously learned support vectors
     * @param[in] alpha the alpha values (weights) associated with the support vectors and classes
     * @param[in] rho the rho values for each class determined after training the model
     * @param[in,out] w the normal vectors to speedup prediction in case of the linear kernel function, an empty vector in case of the polynomial or rbf kernel
     * @param[in] predict_points the points to predict
     * @throws plssvm::exception any exception thrown by the backend's implementation
     * @return a vector filled with the predictions (not the actual labels!) (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<std::vector<float>> predict_values(const detail::parameter<float> &params, const std::vector<std::vector<float>> &support_vectors, const std::vector<std::vector<float>> &alpha, const std::vector<float> &rho, std::vector<std::vector<float>> &w, const std::vector<std::vector<float>> &predict_points) const = 0;
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] virtual std::vector<std::vector<double>> predict_values(const detail::parameter<double> &params, const std::vector<std::vector<double>> &support_vectors, const std::vector<std::vector<double>> &alpha, const std::vector<double> &rho, std::vector<std::vector<double>> &w, const std::vector<std::vector<double>> &predict_points) const = 0;

    /// The target platform of this SVM.
    target_platform target_{ plssvm::target_platform::automatic };
  private:
    /**
     * @brief Perform some sanity checks on the passed SVM parameters.
     * @throws plssvm::invalid_parameter_exception if the kernel function is invalid
     * @throws plssvm::invalid_parameter_exception if the gamma value for the polynomial or radial basis function kernel is **not** greater than zero
     */
    void sanity_check_parameter() const;

    /// The SVM parameter (e.g., cost, degree, gamma, coef0) currently in use.
    parameter params_{};
};

inline csvm::csvm(parameter params) :
    params_{ params } {
    this->sanity_check_parameter();
}

template <typename... Args>
csvm::csvm(Args &&...named_args) :
    params_{ std::forward<Args>(named_args)... } {
    this->sanity_check_parameter();
}

template <typename... Args, std::enable_if_t<detail::has_only_parameter_named_args_v<Args...>, bool>>
void csvm::set_params(Args &&...named_args) {
    static_assert(sizeof...(Args) > 0, "At least one named parameter mus be given when calling set_params()!");

    // create new parameter struct which is responsible for parsing the named_args
    parameter provided_params{ std::forward<Args>(named_args)... };

    // set the value of params_ if and only if the respective value in provided_params isn't the default value
    if (!provided_params.kernel_type.is_default()) {
        params_.kernel_type = provided_params.kernel_type.value();
    }
    if (!provided_params.gamma.is_default()) {
        params_.gamma = provided_params.gamma.value();
    }
    if (!provided_params.degree.is_default()) {
        params_.degree = provided_params.degree.value();
    }
    if (!provided_params.coef0.is_default()) {
        params_.coef0 = provided_params.coef0.value();
    }
    if (!provided_params.cost.is_default()) {
        params_.cost = provided_params.cost.value();
    }

    // check if the new parameters make sense
    this->sanity_check_parameter();
}

template <typename real_type, typename label_type, typename... Args>
model<real_type, label_type> csvm::fit(const data_set<real_type, label_type> &data, Args &&...named_args) const {
    igor::parser parser{ std::forward<Args>(named_args)... };

    // set default values
    default_value epsilon_val{ default_init<real_type>{ 0.001 } };
    default_value max_iter_val{ default_init<unsigned long long>{ data.num_data_points() } };
    default_value classification_val{ default_init<classification_type>{ classification_type::oaa } };

    // compile time check: only named parameter are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(epsilon, max_iter, classification), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(epsilon)) {
        // get the value of the provided named parameter
        epsilon_val = detail::get_value_from_named_parameter<typename decltype(epsilon_val)::value_type>(parser, epsilon);
        // check if value makes sense
        if (epsilon_val <= static_cast<typename decltype(epsilon_val)::value_type>(0)) {
            throw invalid_parameter_exception{ fmt::format("epsilon must be less than 0.0, but is {}!", epsilon_val) };
        }
    }
    if constexpr (parser.has(max_iter)) {
        // get the value of the provided named parameter
        max_iter_val = detail::get_value_from_named_parameter<typename decltype(max_iter_val)::value_type>(parser, max_iter);
        // check if value makes sense
        if (max_iter_val == static_cast<typename decltype(max_iter_val)::value_type>(0)) {
            throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", max_iter_val) };
        }
    }
    if constexpr (parser.has(classification)) {
        // get the value of the provided named parameter
        classification_val = detail::get_value_from_named_parameter<typename decltype(classification_val)::value_type>(parser, classification);
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

    detail::log(verbosity_level::full | verbosity_level::timing,
                "Using {} ({}) as multi-class classification strategy.\n",
                detail::tracking_entry{ "cg", "classification_type", classification_val.value() },
                classification_type_to_full_string(classification_val.value()));

    if (classification_val.value() == plssvm::classification_type::oaa) {
        // use the one vs. all multi-class classification strategy
        // solve the minimization problem
        std::tie(*csvm_model.alpha_ptr_, *csvm_model.rho_ptr_) = solve_system_of_linear_equations(static_cast<detail::parameter<real_type>>(params), data.data(), *data.y_ptr_, epsilon_val.value(), max_iter_val.value());
    } else if (classification_val.value() == plssvm::classification_type::oao) {
        // use the one vs. one multi-class classification strategy
        // TODO: implement
        throw exception{ "Error: currently not implemented!" };
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Solved minimization problem (r = b - Ax) using the Conjugate Gradient (CG) methode in {}.\n\n",
                detail::tracking_entry{ "cg", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    return csvm_model;
}

template <typename real_type, typename label_type>
std::vector<label_type> csvm::predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const {
    if (model.num_features() != data.num_features()) {
        throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
    }

    // predict values -> num_data_points x num_classes
    const std::vector<std::vector<real_type>> votes = predict_values(static_cast<detail::parameter<real_type>>(model.params_), model.data_.data(), *model.alpha_ptr_, *model.rho_ptr_, *model.w_, data.data());

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(votes.size());

    if (model.num_different_labels() == 2) {
        // use sign in case of binary classification
        #pragma omp parallel for default(none) shared(predicted_labels, votes, model) if (!std::is_same_v<label_type, bool>)
        for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
            // map { 1, -1 } to { 0, 1 }
            const std::size_t idx = std::abs(plssvm::operators::sign(votes[i][0]) - 1) / 2;
            predicted_labels[i] = model.data_.mapping_->get_label_by_mapped_index(idx);
        }
    } else {
        // use voting
        #pragma omp parallel for default(none) shared(predicted_labels, votes, model) if (!std::is_same_v<label_type, bool>)
        for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
            const std::size_t argmax = std::distance(votes[i].cbegin(), std::max_element(votes[i].cbegin(), votes[i].cend()));
            predicted_labels[i] = model.data_.mapping_->get_label_by_mapped_index(argmax);
        }
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
    if ((params_.kernel_type == kernel_function_type::polynomial || params_.kernel_type == kernel_function_type::rbf) && !params_.gamma.is_default() && params_.gamma.value() <= 0.0) {
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0.0, but is {}!", params_.gamma) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

/// @cond Doxygen_suppress
namespace detail {

/**
 * @brief Sets the `value` to `false` since the given type @p T is either not a C-SVM or the C-SVM using the requested backend isn't available.
 * @tparam T the type of the C-SVM
 */
template <typename T>
struct csvm_backend_exists : std::false_type {};

}  // namespace detail
/// @endcond

/**
 * @brief Sets the value of the `value` member to `true` if @p T is a C-SVM using an available backend. Ignores any top-level const, volatile, and reference qualifiers.
 * @tparam T the type of the C-SVM
 */
template <typename T>
struct csvm_backend_exists : detail::csvm_backend_exists<detail::remove_cvref_t<T>> {};

/**
 * @brief Sets the value of the `value` member to `true` if @p T is a C-SVM using an available backend. Ignores any top-level const, volatile, and reference qualifiers.
 */
template <typename T>
constexpr bool csvm_backend_exists_v = csvm_backend_exists<T>::value;

}  // namespace plssvm

#endif  // PLSSVM_CSVM_HPP_
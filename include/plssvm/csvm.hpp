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

#include "plssvm/constants.hpp"         // plssvm::verbose
#include "plssvm/data_set.hpp"          // plssvm::data_set
#include "plssvm/detail/utility.hpp"    // plssvm::detail::to_underlying, plssvm::detail::remove_cvref_t
#include "plssvm/kernel_types.hpp"      // plssvm::kernel_type
#include "plssvm/model.hpp"             // plssvm::model
#include "plssvm/parameter.hpp"         // plssvm::parameter

#include "igor/igor.hpp"  // igor::parser
#include "fmt/core.h"     // fmt::format, fmt::print

#include <chrono>       // std::chrono::{steady_clock, duration_cast}
#include <cstddef>      // std::size_t
#include <cstdio>       // stderr
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v, std::is_convertible_v
#include <utility>      // std::move, std::forward
#include <vector>       // std::vector


namespace plssvm {

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the type of the data
 */
template <typename T>
class csvm {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
    using size_type = std::size_t;

    //*************************************************************************************************************************************//
    //                                          constructors, destructor, and assignment operators                                         //
    //*************************************************************************************************************************************//
    explicit csvm(parameter<real_type> params = {});
    template <typename... Args>
    explicit csvm(kernel_type kernel, Args&&... named_args);

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
     * @brief Default copy-assignment-operator since a virtual destructor has been declared.
     * @return `*this`
     */
    csvm &operator=(const csvm &) = default;
    /**
     * @brief Default move-assignment-operator since a virtual destructor has been declared.
     * @return `*this`
     */
    csvm &operator=(csvm &&) noexcept = default;


    //*************************************************************************************************************************************//
    //                                                              fit model                                                              //
    //*************************************************************************************************************************************//
    template <typename label_type, typename... Args>
    [[nodiscard]] model<real_type, label_type> fit(const data_set<real_type, label_type> &data, Args&&... named_args) const;


    //*************************************************************************************************************************************//
    //                                                          predict and score                                                          //
    //*************************************************************************************************************************************//
    template <typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data);
    template <typename label_type>
    [[nodiscard]] std::vector<real_type> predict_values(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data);

    // calculate the accuracy of the model
    template <typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model);
    // calculate the accuracy of the data_set
    template <typename label_type>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model, const data_set<real_type> &data);


  protected:
    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Solves using a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] params the C-SVM parameters used in the respective kernel functions
     * @param[in] A the matrix of the equation \f$Ax = b\f$
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] q subvector of the least-squares matrix equation
     * @param[in] QA_cost the bottom right matrix entry multiplied by the cost parameter
     * @param[in] eps error tolerance
     * @param[in] max_iter the maximum number of CG iterations
     * @return the alpha values (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::pair<std::vector<real_type>, real_type> solve_system_of_linear_equations(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, real_type eps, size_type max_iter) const = 0;
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] params the C-SVM parameters used in the respective kernel functions
     * @param[in] support_vectors the previously learned support vectors
     * @param[in] alpha the alpha values associated with the support vectors
     * @param[in] rho the rho value determined after training the model
     * @param[in] w the normal vector to speedup prediction in case of the linear kernel function, `nullptr` in case of the polynomial or rbf kernel
     * @param[in] predict_points the points to predict
     * @return a [`std::vector<real_type>`](https://en.cppreference.com/w/cpp/container/vector) filled with negative values for each prediction for a data point with the negative class and positive values otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<real_type> predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const = 0;

  private:
    /**
     * @brief Perform some sanity checks on the passed C-SVM parameters.
     */
    void sanity_check_parameter();

    parameter<real_type> params_{};
};


/******************************************************************************
 *             constructors, destructor, and assignment operators             *
 ******************************************************************************/
template <typename T>
csvm<T>::csvm(parameter<real_type> params) : params_{ std::move(params) } {
    this->sanity_check_parameter();
}

template <typename T> template <typename... Args>
csvm<T>::csvm(kernel_type kernel, Args&&... named_args) {
    igor::parser p{ std::forward<Args>(named_args)... };

    // set kernel type
    params_.kernel = kernel;

    // compile time check: only named parameter are permitted
    static_assert(!p.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!p.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!p.has_other_than(gamma, degree, coef0, cost, sycl_implementation_type, sycl_kernel_invocation_type), "An illegal named parameter has been passed!");

    // shorthand function for emitting a warning if a provided parameter is not used by the current kernel function
    const auto print_warning = [kernel](const std::string_view param_name) {
        fmt::print(stderr, "{} parameter provided, which is not used in the {} kernel ({})!", param_name, kernel, kernel_type_to_math_string(kernel));
    };

    // compile time/runtime check: the values must have the correct types
    if constexpr (p.has(gamma)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(gamma))>, decltype(params_.gamma)>, "gamma must be convertible to a real_type!");
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear) {
            print_warning("gamma");
        }
        // set value
        params_.gamma = static_cast<decltype(params_.gamma)>(p(gamma));
    }
    if constexpr (p.has(degree)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(degree))>, decltype(params_.degree)>, "degree must be convertible to an int!");
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear || kernel == kernel_type::rbf) {
            print_warning("degree");
        }
        // set value
        params_.degree = static_cast<decltype(params_.degree)>(p(degree));
    }
    if constexpr (p.has(coef0)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(coef0))>, decltype(params_.coef0)>, "coef0 must be convertible to a real_type!");
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear || kernel == kernel_type::rbf) {
            print_warning("coef0");
        }
        // set value
        params_.coef0 = static_cast<decltype(params_.coef0)>(p(coef0));
    }
    if constexpr (p.has(cost)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(cost))>, decltype(params_.cost)>, "cost must be convertible to a real_type!");
        // set value
        params_.cost = static_cast<decltype(params_.cost)>(p(cost));
    }

    // check if parameters make sense
    this->sanity_check_parameter();
}


/******************************************************************************
 *                                  fit model                                 *
 ******************************************************************************/
template <typename T> template <typename label_type, typename... Args>
auto csvm<T>::fit(const data_set<real_type, label_type> &data, Args&&... named_args) const -> model<real_type, label_type> {
    igor::parser p{ std::forward<Args>(named_args)... };

    // set default values
    default_value epsilon_val = default_init<real_type>{ 0.001 };
    default_value max_iter_val = default_init<size_type>{ data.num_data_points() };

    // compile time check: only named parameter are permitted
    static_assert(!p.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!p.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!p.has_other_than(epsilon, max_iter), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (p.has(epsilon)) {
        if constexpr (std::is_same_v<detail::remove_cvref_t<decltype(p(epsilon))>, decltype(epsilon)>) {
            // a plssvm::default_value has been provided
            if (!p(epsilon).is_default()) {
                // only override it if it doesn't hold the default value
                epsilon_val = p(epsilon);
            }
        } else {
            // a normal value (not wrapped) has been provided
            // compile time check: the value must have the correct type
            static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(epsilon))>, real_type>, "epsilon must be convertible to a real_type!");
            // set value
            epsilon_val = static_cast<real_type>(p(epsilon));
            // check if value makes sense
            if (epsilon_val <= real_type{ 0.0 }) {
                throw invalid_parameter_exception{ fmt::format("epsilon must be less than 0.0, but is {}!", epsilon_val) };
            }
        }
    }
    if constexpr (p.has(max_iter)) {
        if constexpr (std::is_same_v<detail::remove_cvref_t<decltype(p(max_iter))>, decltype(max_iter_val)>) {
            // a plssvm::default_value has been provided
            if (!p(max_iter).is_default()) {
                // only override it if it doesn't hold the default value
                max_iter_val = p(max_iter);
            }
        } else {
            // a normal value (not wrapped) has been provided
            // compile time check: the value must have the correct type
            static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(max_iter))>, size_type>, "max_iter must be convertible to a size_type!");
            // set value
            max_iter_val = static_cast<size_type>(p(max_iter));
            // check if value makes sense
            if (max_iter_val == size_type{ 0 }) {
                throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", max_iter_val) };
            }
        }
    }

    // start fitting the data set using a C-SVM

    if (!data.has_labels()) {
        throw exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
    }

    // copy parameter and set gamma if necessary
    parameter<real_type> params{ params_ };
    if (params.gamma.is_default()) {
        // no gamma provided -> use default value which depends on the number of features of the data set
        params.gamma = real_type{ 1.0 } / data.num_features();
    }


    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    // create model
    model<real_type, label_type> csvm_model{ params, data };

    // solve the minimization problem
    std::tie(*csvm_model.alpha_ptr_, csvm_model.rho_) = solve_system_of_linear_equations(params, data.data(), data.mapped_labels().value().get(), epsilon_val.value(), max_iter_val.value());


    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Solved minimization problem (r = b - Ax) using the Conjugate Gradient (CG) methode in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    return csvm_model;
}


/******************************************************************************
 *                              predict and score                             *
 ******************************************************************************/
template <typename T> template <typename label_type>
auto csvm<T>::predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) -> std::vector<label_type> {
    if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // predict values
    const std::vector<real_type> predicted_values = predict_values(model, data);

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(predicted_values.size());
    #pragma omp parallel for default(none) shared(predicted_labels, predicted_values, model)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        predicted_labels[i] = model.data_.label_from_mapped_value(plssvm::operators::sign(predicted_values[i]));
    }
    return predicted_labels;
}

template <typename T> template <typename label_type>
auto csvm<T>::predict_values(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) -> std::vector<real_type> {
    if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    const std::chrono::time_point start_time = std::chrono::steady_clock::now();

    const std::vector<real_type> predicted_values = predict_values_impl(model.params_, model.data_.data(), *model.alpha_ptr_, model.rho_, *model.w_, data.data());

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Predicted {} data points in {}.\n", data.num_data_points(), std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
    // forward implementation to derived classes
    return predicted_values;
}

template <typename T> template <typename label_type>
auto csvm<T>::score(const model<real_type, label_type> &model) -> real_type {
    return this->score(model, model.data_);
}

template <typename T> template <typename label_type>
auto csvm<T>::score(const model<real_type, label_type> &model, const data_set<real_type> &data) -> real_type {
    if (!data.has_labels()) {
        throw exception{ "the data set to score must have labels set!" };
    } else if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // predict labels
    const std::vector<label_type> predicted_labels = predict(model, data);
    // correct labels
    const std::vector<label_type>& correct_labels = model.labels().value();

    // calculate the accuracy
    size_type correct{ 0 };
    #pragma omp parallel for reduction(+ : correct) default(none) shared(predicted_labels, correct_labels)
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
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0, but is {}!", params_.gamma) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

}  // namespace plssvm

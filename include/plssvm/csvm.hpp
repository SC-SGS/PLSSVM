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
#include "plssvm/matrix.hpp"                      // plssvm::aos_matrix
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
     * @brief Fit a model using the current SVM on the @p data using the provided multi-class classification strategy.
     * @tparam real_type the type of the data (`float` or `double`)
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @tparam Args the type of the potential additional parameters
     * @param[in] data the data used to train the SVM model
     * @param[in] named_args the potential additional parameters (`epsilon`, `max_iter`, and `classification`)
     * @throws plssvm::invalid_parameter_exception if the provided value for `epsilon` is greater or equal than zero
     * @throws plssvm::invlaid_parameter_exception if the provided maximum number of iterations is less or equal than zero
     * @throws plssvm::invalid_parameter_exception if the training @p data does **not** include labels
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::solve_system_of_linear_equations`
     * @note For binary classification **always** one vs. all is used regardless of the provided parameter!
     * @return the learned model (`[[nodiscard]]`)
     */
    template <typename real_type, typename label_type, typename... Args>
    [[nodiscard]] model<real_type, label_type> fit(const data_set<real_type, label_type> &data, Args &&...named_args);

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
//    [[nodiscard]] virtual std::pair<aos_matrix<float>, std::vector<float>> solve_system_of_linear_equations(const detail::parameter<float> &params, const aos_matrix<float> &A, aos_matrix<float> B, float eps, unsigned long long max_iter) const = 0;
//    /**
//     * @copydoc plssvm::csvm::solve_system_of_linear_equations
//     */
//    [[nodiscard]] virtual std::pair<aos_matrix<double>, std::vector<double>> solve_system_of_linear_equations(const detail::parameter<double> &params, const aos_matrix<double> &A, aos_matrix<double> B, double eps, unsigned long long max_iter) const = 0;
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
    [[nodiscard]] virtual aos_matrix<float> predict_values(const detail::parameter<float> &params, const aos_matrix<float> &support_vectors, const aos_matrix<float> &alpha, const std::vector<float> &rho, aos_matrix<float> &w, const aos_matrix<float> &predict_points) const = 0;
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] virtual aos_matrix<double> predict_values(const detail::parameter<double> &params, const aos_matrix<double> &support_vectors, const aos_matrix<double> &alpha, const std::vector<double> &rho, aos_matrix<double> &w, const aos_matrix<double> &predict_points) const = 0;

    template <typename real_type>
    std::pair<aos_matrix<real_type>, std::vector<real_type>> solve_system_of_linear_equations_impl(const aos_matrix<real_type> &A, aos_matrix<real_type> B, const detail::parameter<real_type> &params, real_type eps, unsigned long long max_cg_iter);


    virtual void setup_data_on_devices(const aos_matrix<float> &A) = 0;
    virtual void setup_data_on_devices(const aos_matrix<double> &A) = 0;

    // TODO: rename
    virtual std::vector<float> generate_q2(const detail::parameter<float> &params, const std::size_t num_rows_reduced, const std::size_t num_features) = 0;
    virtual std::vector<double> generate_q2(const detail::parameter<double> &params, const std::size_t num_rows_reduced, const std::size_t num_features) = 0;

    virtual void assemble_kernel_matrix_explicit(const detail::parameter<float> &params, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<float> &q_red, float QA_cost) = 0;
    virtual void assemble_kernel_matrix_explicit(const detail::parameter<double> &params, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<double> &q_red, double QA_cost) = 0;

    virtual aos_matrix<float> kernel_matrix_matmul_explicit(const aos_matrix<float> &vec) = 0;
    virtual aos_matrix<double> kernel_matrix_matmul_explicit(const aos_matrix<double> &vec) = 0;

    virtual void clear_data_on_devices(float) = 0;
    virtual void clear_data_on_devices(double) = 0;

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
model<real_type, label_type> csvm::fit(const data_set<real_type, label_type> &data, Args &&...named_args) {
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

    detail::log(verbosity_level::full | verbosity_level::timing,
                "Using {} ({}) as multi-class classification strategy.\n",
                classification_val.value(),
                classification_type_to_full_string(classification_val.value()));

    // create model
    model<real_type, label_type> csvm_model{ params, data, classification_val.value() };


    if (classification_val.value() == plssvm::classification_type::oaa) {
        // use the one vs. all multi-class classification strategy
        // solve the minimization problem
        aos_matrix<real_type> alpha;
        std::tie(alpha, *csvm_model.rho_ptr_) = solve_system_of_linear_equations_impl(*data.data_ptr_, *data.y_ptr_, static_cast<detail::parameter<real_type>>(params), epsilon_val.value(), max_iter_val.value());
        csvm_model.alpha_ptr_->push_back(std::move(alpha));
    } else if (classification_val.value() == plssvm::classification_type::oao) {
        // use the one vs. one multi-class classification strategy
        const std::size_t num_classes = data.num_classes();
        const std::size_t num_binary_classifications = calculate_number_of_classifiers(classification_type::oao, num_classes);
        const std::size_t num_features = data.num_features();
        // resize alpha_ptr_ and rho_ptr_ to the correct sizes
        csvm_model.alpha_ptr_->resize(num_binary_classifications);
        csvm_model.rho_ptr_->resize(num_binary_classifications);

        // create index vector: indices[0] contains the indices of all data points in the big data set with label index 0, and so on
        std::vector<std::vector<std::size_t>> indices(num_classes);
        {
            const std::vector<label_type>& labels = data.labels().value();
            for (std::size_t i = 0; i < data.num_data_points(); ++i) {
                indices[data.mapping_->get_mapped_index_by_label(labels[i])].push_back(i);
            }
        }

        if (num_classes == 2) {
            // special optimization for binary case (no temporary copies necessary)
            detail::log(verbosity_level::full | verbosity_level::timing,
                        "\nClassifying 0 vs 1 ({} vs {}) (1/1):\n",
                        data.mapping_->get_label_by_mapped_index(0),
                        data.mapping_->get_label_by_mapped_index(1));
            const auto &[alpha, rho] = solve_system_of_linear_equations_impl(*data.data_ptr_, *data.y_ptr_, static_cast<detail::parameter<real_type>>(params), epsilon_val.value(), max_iter_val.value());
            csvm_model.alpha_ptr_->front() = std::move(alpha);
            csvm_model.rho_ptr_->front() = rho.front();  // prevents std::tie
        } else {
            // perform one vs. one classification
            std::size_t pos = 0;
            for (std::size_t i = 0; i < num_classes; ++i) {
                for (std::size_t j = i + 1; j < num_classes; ++j) {
                    // TODO: reduce amount of copies!?
                    // assemble one vs. one classification matrix and rhs
                    const std::size_t num_data_points_in_sub_matrix{ indices[i].size() + indices[j].size() };
                    aos_matrix<real_type> binary_data{ num_data_points_in_sub_matrix, num_features };
                    aos_matrix<real_type> binary_y{ 1, num_data_points_in_sub_matrix };  // note: the first dimension will always be one, since only one rhs is needed

                    // note: if this is changed, it must also be changed in the libsvm_model_parsing.hpp in the calculate_alpha_idx function!!!
                    // order the indices in increasing order
                    std::vector<std::size_t> sorted_indices(num_data_points_in_sub_matrix);
                    std::merge(indices[i].cbegin(), indices[i].cend(), indices[j].cbegin(), indices[j].cend(), sorted_indices.begin());
                    // copy the data points to the binary data set
                    #pragma omp parallel for default(none) shared(sorted_indices, binary_data, binary_y, data, indices) firstprivate(num_data_points_in_sub_matrix, num_features, i)
                    for (std::size_t si = 0; si < num_data_points_in_sub_matrix; ++si) {
                        for (std::size_t dim = 0; dim < num_features; ++dim) {
                            binary_data(si, dim) = (*data.data_ptr_)(sorted_indices[si], dim);
                        }
                        // needs only the check against i, since sorted_indices is guaranteed to only contain indices from i and j
                        binary_y(0, si) = detail::contains(indices[i], sorted_indices[si]) ? real_type{ 1.0 } : real_type{ -1.0 };
                    }

                    // if max_iter is the default value, update it according to the current binary classification matrix size
                    const unsigned long long binary_max_iter = max_iter_val.is_default() ? static_cast<unsigned long long>(binary_data.num_rows()) : max_iter_val.value();
                    // solve the minimization problem -> note that only a single rhs is present
                    detail::log(verbosity_level::full | verbosity_level::timing,
                                "\nClassifying {} vs {} ({} vs {}) ({}/{}):\n",
                                i,
                                j,
                                data.mapping_->get_label_by_mapped_index(i),
                                data.mapping_->get_label_by_mapped_index(j),
                                pos + 1,
                                calculate_number_of_classifiers(classification_type::oao, num_classes));
                    const auto &[alpha, rho] = solve_system_of_linear_equations_impl(binary_data, binary_y, static_cast<detail::parameter<real_type>>(params), epsilon_val.value(), binary_max_iter);
                    (*csvm_model.alpha_ptr_)[pos] = std::move(alpha);
                    (*csvm_model.rho_ptr_)[pos] = rho.front();  // prevents std::tie
                    // go to next one vs. one classification
                    ++pos;
                    // order of the alpha value: 0 vs 1, 0 vs 2, 0 vs 3, 1 vs 2, 1 vs 3, 2 vs 3
                }
            }
        }

        csvm_model.indices_ptr_ = std::make_shared<typename decltype(csvm_model.indices_ptr_)::element_type>(std::move(indices));
    }

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "\nLearned the SVM classifier for {} multi-class classification in {}.\n\n",
                classification_type_to_full_string(classification_val.value()),
                detail::tracking_entry{ "cg", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    return csvm_model;
}

template <typename real_type, typename label_type>
std::vector<label_type> csvm::predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) const {
    if (model.num_features() != data.num_features()) {
        throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
    }

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(data.num_data_points());

    PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (predict points) may never be a nullptr!");
    const aos_matrix<real_type> &predict_points = *data.data_ptr_;

    if (model.get_classification_type() == classification_type::oaa) {
        PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (model) may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_ != nullptr, "The alpha_ptr_ may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_->size() == 1, "For OAA, the alpha vector must only contain a single aos_matrix of size {}x{}!", model.num_classes(), model.num_support_vectors());
        PLSSVM_ASSERT(model.alpha_ptr_->front().num_rows() == calculate_number_of_classifiers(classification_type::oaa, data.num_classes()), "The number of rows in the matrix must be {}, but is {}!", model.alpha_ptr_->front().num_rows(), calculate_number_of_classifiers(classification_type::oaa, data.num_classes()));
        PLSSVM_ASSERT(model.alpha_ptr_->front().num_cols() == data.num_data_points(), "The number of weights ({}) must be equal to the number of support vectors ({})!", model.alpha_ptr_->front().num_cols(), data.num_data_points());

        const aos_matrix<real_type> &sv = *model.data_.data_ptr_;
        const aos_matrix<real_type> &alpha = model.alpha_ptr_->back();

        // predict values using OAA -> num_data_points x num_classes
        const aos_matrix<real_type> votes = predict_values(static_cast<detail::parameter<real_type>>(model.params_), sv, alpha, *model.rho_ptr_, *model.w_ptr_, predict_points);

        PLSSVM_ASSERT(votes.num_rows() == data.num_data_points(), "The number of votes ({}) must be equal the number of data points ({})!", votes.num_rows(), data.num_data_points());
        PLSSVM_ASSERT(votes.num_cols() == calculate_number_of_classifiers(classification_type::oaa, model.num_classes()), "The votes contain {} values, but must contain {} values!", votes.num_cols(), calculate_number_of_classifiers(classification_type::oaa, model.num_classes()));

        // use voting
        #pragma omp parallel for default(none) shared(predicted_labels, votes, model) if (!std::is_same_v<label_type, bool>)
        for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
            std::size_t argmax = 0;
            real_type max = std::numeric_limits<real_type>::lowest();
            for (std::size_t v = 0; v < votes.num_cols(); ++v) {
                if (max < votes(i, v)) {
                    argmax = v;
                    max = votes(i, v);
                }
            }
            predicted_labels[i] = model.data_.mapping_->get_label_by_mapped_index(argmax);
        }
    } else if (model.get_classification_type() == classification_type::oao) {
        PLSSVM_ASSERT(model.indices_ptr_ != nullptr, "The indices_ptr_ may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_ != nullptr, "The alpha_ptr_ may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_->size() == calculate_number_of_classifiers(classification_type::oao, model.num_classes()), "The alpha vector must contain {} matrices, but it contains {}!", calculate_number_of_classifiers(classification_type::oao, model.num_classes()), model.alpha_ptr_->size());
        PLSSVM_ASSERT(std::all_of(model.alpha_ptr_->cbegin(), model.alpha_ptr_->cend(), [](const aos_matrix<real_type> &matr) { return matr.num_rows() == 1; }), "In case of OAO, each matrix may only contain one row!");
        PLSSVM_ASSERT(model.rho_ptr_ != nullptr, "The rho_ptr_ may never be a nullptr!");
        PLSSVM_ASSERT(model.w_ptr_ != nullptr, "The w_ptr_ may never be a nullptr!");

        // predict values using OAO
        const std::size_t num_features = model.num_features();
        const std::size_t num_classes = model.num_classes();
        const std::vector<std::vector<std::size_t>> &indices = *model.indices_ptr_;

        aos_matrix<std::size_t> class_votes{ data.num_data_points(), num_classes };

        bool calculate_w{ false };
        if (model.w_ptr_->empty()) {
            // w is currently empty
            // initialize the w matrix and calculate it later!
            calculate_w = true;
            (*model.w_ptr_) = aos_matrix<real_type>{ calculate_number_of_classifiers(classification_type::oao, num_classes), num_features };
        }

        // perform one vs. one prediction
        std::size_t pos = 0;
        for (std::size_t i = 0; i < num_classes; ++i) {
            for (std::size_t j = i + 1; j < num_classes; ++j) {
                // TODO: reduce amount of copies!?
                // assemble one vs. one classification matrix and rhs
                const std::size_t num_data_points_in_sub_matrix{ indices[i].size() + indices[j].size() };
                const aos_matrix<real_type> &binary_alpha = (*model.alpha_ptr_)[pos];
                const std::vector<real_type> binary_rho{ (*model.rho_ptr_)[pos] };

                // create binary support vector matrix, based on the number of classes
                const aos_matrix<real_type> &binary_sv = [&]() {
                   if (num_classes == 2) {
                       // no special assembly needed in binary case
                       return *model.data_.data_ptr_;
                   } else {
                       // note: if this is changed, it must also be changed in the libsvm_model_parsing.hpp in the calculate_alpha_idx function!!!
                       // order the indices in increasing order
                       aos_matrix<real_type> temp{ num_data_points_in_sub_matrix, num_features };
                       std::vector<std::size_t> sorted_indices(num_data_points_in_sub_matrix);
                       std::merge(indices[i].cbegin(), indices[i].cend(), indices[j].cbegin(), indices[j].cend(), sorted_indices.begin());
                        // copy the support vectors to the binary support vectors
                        #pragma omp parallel for collapse(2) default(none) shared(sorted_indices, temp, model) firstprivate(num_data_points_in_sub_matrix, num_features)
                       for (std::size_t si = 0; si < num_data_points_in_sub_matrix; ++si) {
                           for (std::size_t dim = 0; dim < num_features; ++dim) {
                               temp(si, dim) = (*model.data_.data_ptr_)(sorted_indices[si], dim);
                           }
                       }
                       return temp;
                   }
                }();

                // predict binary pair
                aos_matrix<real_type> binary_votes{};
                // don't use the w vector for the polynomial and rbf kernel OR if the w vector hasn't been calculated yet
                if (params_.kernel_type != kernel_function_type::linear || calculate_w) {
                    // the w vector optimization has not been applied yet -> calculate w and store it
                    aos_matrix<real_type> w{};
                    // returned w: 1 x num_features
                    binary_votes = predict_values(static_cast<detail::parameter<real_type>>(model.params_), binary_sv, binary_alpha, binary_rho, w, predict_points);
                    // only in case of the linear kernel, the w vector gets filled -> store it
                    if (params_.kernel_type == kernel_function_type::linear) {
                        #pragma omp parallel for default(none) shared(model, w) firstprivate(num_features, pos)
                        for (std::size_t dim = 0; dim < num_features; ++dim) {
                            (*model.w_ptr_)(pos, dim) = w(0, dim);
                        }
                    }
                } else {
                    // use previously calculated w vector
                    aos_matrix<real_type> binary_w{ 1, num_features };
                    #pragma omp parallel for default(none) shared(model, binary_w) firstprivate(num_features, pos)
                    for (std::size_t dim = 0; dim < num_features; ++dim) {
                        binary_w(0, dim) = (*model.w_ptr_)(pos, dim);
                    }
                    binary_votes = predict_values(static_cast<detail::parameter<real_type>>(model.params_), binary_sv, binary_alpha, binary_rho, binary_w, predict_points);
                }

                PLSSVM_ASSERT(binary_votes.num_rows() == data.num_data_points(), "The number of votes ({}) must be equal the number of data points ({})!", binary_votes.num_rows(), data.num_data_points());
                PLSSVM_ASSERT(binary_votes.num_cols() == 1, "The votes contain {} values, but must contain one value!", binary_votes.num_cols());

                #pragma omp parallel for default(none) shared(data, binary_votes, class_votes) firstprivate(i, j)
                for (std::size_t d = 0; d < data.num_data_points(); ++d) {
                    if (binary_votes(d, 0) > real_type{ 0.0 }) {
                        ++class_votes(d, i);
                    } else {
                        ++class_votes(d, j);
                    }
                }

                // go to next one vs. one classification
                ++pos;
                // order of the alpha value: 0 vs 1, 0 vs 2, 0 vs 3, 1 vs 2, 1 vs 3, 2 vs 3
            }
        }

        // map majority vote to predicted class
        #pragma omp parallel for default(none) shared(predicted_labels, class_votes, model) if (!std::is_same_v<label_type, bool>)
        for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
            std::size_t argmax = 0;
            real_type max = std::numeric_limits<real_type>::lowest();
            for (std::size_t v = 0; v < class_votes.num_cols(); ++v) {
                if (max < class_votes(i, v)) {
                    argmax = v;
                    max = class_votes(i, v);
                }
            }
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
    #pragma omp parallel for default(none) shared(predicted_labels, correct_labels) reduction(+ : correct)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == correct_labels[i]) {
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(predicted_labels.size());
}

void csvm::sanity_check_parameter() const {
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

// TODO: move to correct place!
template <typename real_type>
std::pair<aos_matrix<real_type>, std::vector<real_type>> csvm::solve_system_of_linear_equations_impl(const aos_matrix<real_type> &A, aos_matrix<real_type> B_in, const detail::parameter<real_type> &params, real_type eps, unsigned long long max_cg_iter) {
    using namespace plssvm::operators;
    using clock_type = std::chrono::steady_clock;

//    const std::size_t num_rows = A.num_rows();
    const std::size_t num_features = A.num_cols();
    const std::size_t num_rows_reduced = A.num_rows() - 1;
    const std::size_t num_rhs = B_in.num_rows();

    // setup/allocate necessary data on the device(s)
    this->setup_data_on_devices(A);

    const clock_type::time_point dimension_reduction_start_time = clock_type::now();
    // create q_red vector and calculate QA_costs
    const std::vector<real_type> q_red = this->generate_q2(params, num_rows_reduced, num_features);
    const clock_type::time_point dimension_reduction_end_time = clock_type::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Performed dimensional reduction in {}.\n",
                detail::tracking_entry{ "cg", "dimensional_reduction", std::chrono::duration_cast<std::chrono::milliseconds>(dimension_reduction_end_time - dimension_reduction_start_time) });
    const real_type QA_cost = kernel_function(A, num_rows_reduced, A, num_rows_reduced, params) + real_type{ 1.0 } / params.cost;

    // update right-hand sides (B)
    std::vector<real_type> b_back_value(num_rhs);
    aos_matrix<real_type> B{ num_rhs, num_rows_reduced };
    #pragma omp parallel for default(none) shared(B_in, B, b_back_value) firstprivate(num_rhs, num_rows_reduced)
    for (std::size_t row = 0; row < num_rhs; ++row) {
        b_back_value[row] = B_in(row, num_rows_reduced);
        for (std::size_t col = 0; col < num_rows_reduced; ++col) {
            B(row, col) = B_in(row, col) - b_back_value[row];
        }
    }

    // assemble explicit kernel matrix
    {
        const clock_type::time_point assembly_start_time = clock_type::now();
        this->assemble_kernel_matrix_explicit(params, num_rows_reduced, num_features, q_red, QA_cost);
        const clock_type::time_point assembly_end_time = clock_type::now();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Assembled the kernel matrix in {}.\n",
                    detail::tracking_entry{ "cg", "kernel_matrix_assembly", std::chrono::duration_cast<std::chrono::milliseconds>(assembly_end_time - assembly_start_time) });
    }

    //
    // perform Conjugate Gradients (CG) algorithm
    //

    aos_matrix<real_type> X{ num_rhs, num_rows_reduced, real_type{ 1.0 }};

    // R = B - A * X
    aos_matrix<real_type> R = this->kernel_matrix_matmul_explicit(X);
    #pragma omp parallel for collapse(2) shared(R, B) firstprivate(num_rhs, num_rows_reduced)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        for (std::size_t j = 0; j < num_rows_reduced; ++j) {
            R(i, j) = B(i, j) - R(i, j);
        }
    }

    // delta = R.T * R
    std::vector<real_type> delta(num_rhs);
    #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, num_rows_reduced)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (std::size_t j = 0; j < num_rows_reduced; ++j) {
            temp += R(i, j) * R(i, j);
        }
        delta[i] = temp;
    }
    const std::vector<real_type> delta0(delta);

    aos_matrix<real_type> D{ R };


    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        const auto iteration_end_time = std::chrono::steady_clock::now();
        const auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n",
                    iteration_duration);
        average_iteration_time += iteration_duration;
    };
    // get the index of the rhs that has the largest residual difference wrt to its target residual
    const auto residual_info = [&]() {
        real_type max_difference{ 0.0 };
        std::size_t idx{ 0 };
        for (std::size_t i = 0; i < delta.size(); ++i) {
            const real_type difference = delta[i] - (eps * eps * delta0[i]);
            if (difference > max_difference) {
                idx = i;
            }
        }
        return idx;
    };
    // get the number of rhs that have already been converged
    const auto num_converged = [&]() {
        std::vector<bool> converged(delta.size(), false);
        for (std::size_t i = 0; i < delta.size(); ++i) {
            // check if the rhs converged in the current iteration
            converged[i] = delta[i] <= eps * eps * delta0[i];
        }
        return static_cast<std::size_t>(std::count(converged.cbegin(), converged.cend(), true));
    };

    unsigned long long iter = 0;
    for (; iter < max_cg_iter; ++iter) {
        const std::size_t max_residual_difference_idx = residual_info();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with {}/{} converged rhs (max residual {} with target residual {} for rhs {}). ",
                    iter + 1,
                    max_cg_iter,
                    num_converged(),
                    num_rhs,
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        iteration_start_time = std::chrono::steady_clock::now();

        // Q = A * D
        const aos_matrix<real_type> Q = this->kernel_matrix_matmul_explicit(D);

        // alpha = delta_new / (D^T * Q))
        std::vector<real_type> alpha(num_rhs);
        #pragma omp parallel for default(none) shared(D, Q, alpha, delta) firstprivate(num_rhs, num_rows_reduced)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < num_rows_reduced; ++dim) {
                temp += D(i, dim) * Q(i, dim);
            }
            alpha[i] = delta[i] / temp;
        }

        // X = X + alpha * D
        #pragma omp parallel for collapse(2) default(none) shared(X, alpha, D) firstprivate(num_rhs, num_rows_reduced)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < num_rows_reduced; ++dim) {
                X(i, dim) += alpha[i] * D(i, dim);
            }
        }

        if (iter % 50 == 49) {
            // explicitly recalculate residual to remove accumulating floating point errors
            // R = B - A * X
            R = this->kernel_matrix_matmul_explicit(X);
            #pragma omp parallel for collapse(2) shared(R, B) firstprivate(num_rhs, num_rows_reduced)
            for (std::size_t i = 0; i < num_rhs; ++i) {
                for (std::size_t j = 0; j < num_rows_reduced; ++j) {
                    R(i, j) = B(i, j) - R(i, j);
                }
            }
        } else {
            // R = R - alpha * Q
            #pragma omp parallel for collapse(2) default(none) shared(R, alpha, Q) firstprivate(num_rhs, num_rows_reduced)
            for (std::size_t i = 0; i < num_rhs; ++i) {
                for (std::size_t dim = 0; dim < num_rows_reduced; ++dim) {
                    R(i, dim) -= alpha[i] * Q(i, dim);
                }
            }
        }

        // delta = R^T * R
        const std::vector<real_type> delta_old = delta;
        #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, num_rows_reduced)
        for (std::size_t col = 0; col < num_rhs; ++col) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t row = 0; row < num_rows_reduced; ++row) {
                temp += R(col, row) * R(col, row);
            }
            delta[col] = temp;
        }

        // if we are exact enough stop CG iterations, i.e., if every rhs has converged
        if (num_converged() == num_rhs) {
            output_iteration_duration();
            break;
        }

        // beta = delta_new / delta_old
        const std::vector<real_type> beta = delta / delta_old;
        // D = beta * D + R
        #pragma omp parallel for collapse(2) default(none) shared(D, beta, R) firstprivate(num_rhs, num_rows_reduced)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < num_rows_reduced; ++dim) {
                D(i, dim) = beta[i] * D(i, dim) + R(i, dim);
            }
        }

        output_iteration_duration();
    }
    const std::size_t max_residual_difference_idx = residual_info();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", std::min(iter + 1, max_cg_iter) },
                detail::tracking_entry{ "cg", "max_iterations", max_cg_iter },
                detail::tracking_entry{ "cg", "num_converged_rhs", num_converged() },
                detail::tracking_entry{ "cg", "num_rhs", num_rhs },
                delta[max_residual_difference_idx],
                eps * eps * delta0[max_residual_difference_idx],
                max_residual_difference_idx,
                detail::tracking_entry{ "cg", "avg_iteration_time", average_iteration_time / std::min(iter + 1, max_cg_iter) });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "residuals", delta }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "target_residuals", eps * eps * delta0 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "epsilon", eps }));
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n",
                std::min(iter + 1, max_cg_iter));


    // calculate bias
    aos_matrix<real_type> X_ret{ num_rhs, A.num_rows() };
    std::vector<real_type> bias(num_rhs);
    #pragma omp parallel for default(none) shared(X, q_red, X_ret, bias, b_back_value) firstprivate(num_rhs, num_rows_reduced, QA_cost)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp_sum{ 0.0 };
        real_type temp_dot{ 0.0 };
        #pragma omp simd reduction(+ : temp_sum) reduction(+ : temp_dot)
        for (std::size_t dim = 0; dim < num_rows_reduced; ++dim) {
            temp_sum += X(i, dim);
            temp_dot += q_red[dim] * X(i, dim);

            X_ret(i, dim) = X(i, dim);
        }
        bias[i] = -(b_back_value[i] + QA_cost * temp_sum - temp_dot);
        X_ret(i, num_rows_reduced) = -temp_sum;
    }

    this->clear_data_on_devices(real_type{});

    return std::make_pair(std::move(X_ret), std::move(bias));
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
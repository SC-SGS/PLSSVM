/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them for the regression task.
 */

#ifndef PLSSVM_SVM_CSVR_HPP_
#define PLSSVM_SVM_CSVR_HPP_
#pragma once

#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/data_set/regression_data_set.hpp"         // plssvm::regression_data_set
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/logging/mpi_log.hpp"               // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::check_local_memory_usage
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::invalid_parameter_exception, plssvm::mpi_exception
#include "plssvm/kernel_function_types.hpp"                // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                               // plssvm::aos_matrix
#include "plssvm/model/regression_model.hpp"               // plssvm::regression_model
#include "plssvm/parameter.hpp"                            // plssvm::parameter
#include "plssvm/regression_report.hpp"                    // plssvm::regression_report
#include "plssvm/svm/csvm.hpp"                             // plssvm::csvm
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "fmt/format.h"   // fmt::format
#include "igor/igor.hpp"  // igor::parser

#include <algorithm>    // std::all_of
#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cmath>        // std::round
#include <cstddef>      // std::size_t
#include <memory>       // std::addressof
#include <optional>     // std::make_optional
#include <tuple>        // std::tie
#include <type_traits>  // std::is_floating_point_v
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm {

/**
 * @example csvr_examples.cpp
 * @brief A few examples regarding the plssvm::csvr class.
 */

/**
 * @brief Base class for all C-SVR backends.
 * @details This class implements all features shared between all C-SVR backends. It defines the whole public API of a C-SVR.
 */
class csvr : virtual public csvm {
  public:
    /// The type of the model returned by a call to the `fit` function and used in the `predict` and `score` functions.
    template <typename T>
    using model_type = ::plssvm::regression_model<T>;

    // inherit C-SVM base class constructors
    using ::plssvm::csvm::csvm;

    /**
     * @copydoc plssvm::csvm::csvm(const plssvm::csvm &)
     */
    csvr(const csvr &) = delete;
    /**
     * @copydoc plssvm::csvm::csvm(plssvm::csvm &&) noexcept
     */
    csvr(csvr &&) noexcept = default;
    /**
     * @copydoc plssvm::csvm::operator=(const plssvm::csvm &)
     */
    csvr &operator=(const csvr &) = delete;

    /**
     * @brief Correctly implement the move-assignment operator in presence of a virtual base class.
     * @details Calls the base class move-assignment operator. Afterwards, moves the potential additional `csvr` members.
     * @param[in,out] other the other C-SVM to move from
     * @return `*this`
     */
    csvr &operator=(csvr &&other) noexcept {
        if (this != std::addressof(other)) {
            ::plssvm::csvm::operator=(std::move(other));
        }
        return *this;
    }

    /**
     * @copydoc plssvm::csvm::~csvm() noexcept
     */
    ~csvr() noexcept override = default;

    //*************************************************************************************************************************************//
    //                                                              fit model                                                              //
    //*************************************************************************************************************************************//
    /**
     * @brief Fit a model using the current SVM on the @p data using the provided multi-class classification strategy.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @tparam Args the type of the potential additional parameters
     * @param[in] data the data used to train the SVM model
     * @param[in] named_args the potential additional parameters (`epsilon`, `max_iter`, and `classification`)
     * @throws plssvm::invalid_parameter_exception if the provided value for `epsilon` is greater or equal than zero
     * @throws plssvm::invlaid_parameter_exception if the provided maximum number of iterations is less or equal than zero
     * @throws plssvm::invalid_parameter_exception if the training @p data does **not** include labels
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::solve_lssvm_system_of_linear_equations`
     * @throws plssvm::mpi_exception if the MPI communicator of the C-SVR and the MPI communicator of the @p data set are not identical
     * @note For binary classification **always** one vs. all is used regardless of the provided parameter!
     * @return the learned model (`[[nodiscard]]`)
     */
    template <typename label_type, typename... Args>
    [[nodiscard]] regression_model<label_type> fit(const regression_data_set<label_type> &data, Args &&...named_args) const {
#if defined(PLSSVM_ENABLE_ASSERTS)
        if (params_.kernel_type == kernel_function_type::chi_squared) {
            PLSSVM_ASSERT(std::all_of(data.data().data(), data.data().data() + data.data().size(), [](const real_type val) { return val >= real_type{ 0.0 }; }),
                          "The chi-squared kernel is only well defined for non-negative values!");
        }
#endif

        if (!data.has_labels()) {
            throw invalid_parameter_exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
        }
        // check whether the C-SVR and data set MPI communicators are identical
        if (comm_ != data.communicator()) {
            throw mpi_exception{ "The MPI communicators provided to the C-SVR and data set must be identical!" };
        }

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("fit start");

        const igor::parser parser{ named_args... };

        // compile time check: only named parameters are permitted
        static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
        // compile time check: each named parameter must only be passed once
        static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
        // compile time check: only some named parameters are allowed
        static_assert(!parser.has_other_than(epsilon, max_iter, solver), "An illegal named parameter has been passed!");

        // start fitting the data set using a C-SVM
        const std::chrono::time_point start_time = std::chrono::steady_clock::now();

        // copy parameter and set gamma if necessary
        parameter params{ params_ };
        // if the active gamma_type variant member isn't a real_type, replace it with a real_type value by calculating its true value based on the used data set
        // -> params.gamma is guaranteed to be a real_type now!
        params.gamma = calculate_gamma_value(params_.gamma, data.data());

        // create regression model
        regression_model<label_type> csvr_model{ params, data };
        std::vector<unsigned long long> num_iters{};

        // solve the minimization problem
        aos_matrix<real_type> alpha{};
        std::tie(alpha, *csvr_model.rho_ptr_, num_iters) = this->solve_lssvm_system_of_linear_equations(*data.data_ptr_, *data.y_ptr_, params, std::forward<Args>(named_args)...);
        csvr_model.alpha_ptr_->push_back(std::move(alpha));

        // move number of CG iterations to model
        csvr_model.num_iters_ = std::make_optional(std::move(num_iters));

        const std::chrono::time_point end_time = std::chrono::steady_clock::now();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    comm_,
                    "\nLearned the SVR classifier for regression in {}.\n\n",
                    detail::tracking::tracking_entry{ "cg", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("fit end");

        return csvr_model;
    }

    //*************************************************************************************************************************************//
    //                                                          predict and score                                                          //
    //*************************************************************************************************************************************//
    /**
     * @brief Predict the labels for the @p data set using the @p model.
     * @tparam label_type the type of the label
     * @param[in] model a previously learned model
     * @param[in] data the data to predict the labels for
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @throws plssvm::mpi_exception if the MPI communicator of the C-SVR and the MPI communicator of the @p model set are not identical
     * @throws plssvm::mpi_exception if the MPI communicator of the C-SVR and the MPI communicator of the @p data set are not identical
     * @return the predicted labels (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const regression_model<label_type> &model, const regression_data_set<label_type> &data) const {
#if defined(PLSSVM_ENABLE_ASSERTS)
        if (params_.kernel_type == kernel_function_type::chi_squared) {
            PLSSVM_ASSERT(std::all_of(data.data().data(), data.data().data() + data.data().size(), [](const real_type val) { return val >= real_type{ 0.0 }; }),
                          "The chi-squared kernel is only well defined for non-negative values!");
        }
#endif

        if (model.num_features() != data.num_features()) {
            throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
        }
        // check whether the C-SVR and model MPI communicators are identical
        if (comm_ != model.communicator()) {
            throw mpi_exception{ "The MPI communicators provided to the C-SVR and model must be identical!" };
        }
        // check whether the C-SVR and data set MPI communicators are identical
        if (comm_ != data.communicator()) {
            throw mpi_exception{ "The MPI communicators provided to the C-SVR and data set must be identical!" };
        }

        // determine the used local memory and check whether it exceeds the maximum necessary value!
        detail::check_local_memory_usage(this->get_local_memory());

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("predict start");

        // convert predicted values to the correct labels
        std::vector<label_type> predicted_labels(data.num_data_points());

        PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (predict points) may never be a nullptr!");
        const aos_matrix<real_type> &predict_points = *data.data_ptr_;

        PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (model) may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_ != nullptr, "The alpha_ptr_ may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_->size() == 1, "The alpha vector must only contain a single aos_matrix of size {}x{}!", 1, model.num_support_vectors());
        PLSSVM_ASSERT(model.alpha_ptr_->front().num_rows() == 1, "The number of rows in the matrix must be exactly one, but is {}!", model.alpha_ptr_->front().num_rows());

        const aos_matrix<real_type> &sv = model.support_vectors();
        const aos_matrix<real_type> &alpha = model.alpha_ptr_->front();  // num_classes x num_data_points

        // predict values
        const aos_matrix<real_type> votes = this->run_predict_values(model.params_, sv, alpha, *model.rho_ptr_, *model.w_ptr_, predict_points);

        PLSSVM_ASSERT(votes.num_rows() == data.num_data_points(), "The number of votes ({}) must be equal the number of data points ({})!", votes.num_rows(), data.num_data_points());
        PLSSVM_ASSERT(votes.num_cols() == 1, "The votes contain {} values, but must contain exactly one value!", votes.num_cols());

        for (std::size_t i = 0; i < data.num_data_points(); ++i) {
            // TODO: is there multiclass regression? https://en.wikipedia.org/wiki/Multinomial_logistic_regression
            if constexpr (std::is_floating_point_v<label_type>) {
                predicted_labels[i] = static_cast<label_type>(votes(i, 0));
            } else {
                predicted_labels[i] = static_cast<label_type>(std::round(votes(i, 0)));
            }
        }

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("predict end");

        return predicted_labels;
    }

    /**
     * @brief Calculate the regression loss of the @p model using the coefficient of determination.
     * @details A model read from a LIBSVM model file can't be directly fitted, since it doesn't contain the original label information.
     * @tparam label_type the type of the label
     * @param[in] model a previously learned model
     * @throws plssvm::invalid_parameter_exception if the @p model has no labels
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @throws plssvm::mpi_exception if the MPI communicator of the C-SVR and the MPI communicator of the @p model set are not identical
     * @return the regression loss of the model (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type
    score(const regression_model<label_type> &model) const {
        if (!model.data_->has_labels()) {
            throw invalid_parameter_exception{ "The model must have labels to score it! Maybe to model was read from a LIBSVM model file?" };
        }
        return this->score(model, dynamic_cast<const regression_data_set<label_type> &>(*model.data_));
    }

    /**
     * @brief Calculate the regression loss of the labeled @p data set using the @p model using the coefficient of determination.
     * @tparam label_type the type of the label
     * @param[in] model a previously learned model
     * @param[in] data the labeled data set to score
     * @throws plssvm::invalid_parameter_exception if the @p data to score has no labels
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @throws plssvm::mpi_exception if the MPI communicator of the C-SVR and the MPI communicator of the @p model set are not identical
     * @throws plssvm::mpi_exception if the MPI communicator of the C-SVR and the MPI communicator of the @p data set are not identical
     * @return the regression loss of the labeled @p data (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const regression_model<label_type> &model, const regression_data_set<label_type> &data) const {
        // the data set must contain labels in order to score the learned model
        const std::optional<std::vector<label_type>> &correct_labels_opt = data.labels();
        if (!correct_labels_opt.has_value()) {
            throw invalid_parameter_exception{ "The data set to score must have labels!" };
        }
        // the number of features must be equal
        if (model.num_features() != data.num_features()) {
            throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
        }
        // check whether the C-SVR and model MPI communicators are identical
        if (comm_ != model.communicator()) {
            throw mpi_exception{ "The MPI communicators provided to the C-SVR and model must be identical!" };
        }
        // check whether the C-SVR and data set MPI communicators are identical
        if (comm_ != data.communicator()) {
            throw mpi_exception{ "The MPI communicators provided to the C-SVR and data set must be identical!" };
        }

        // predict labels
        const std::vector<label_type> predicted_labels = this->predict(model, data);
        // correct labels
        const std::vector<label_type> &correct_labels = correct_labels_opt.value();

        return regression_report{ correct_labels, predicted_labels }.loss().r2_score;
    }
};

}  // namespace plssvm

#endif  // PLSSVM_SVM_CSVR_HPP_

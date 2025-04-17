/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them for the classification task.
 */

#ifndef PLSSVM_SVM_CSVC_HPP_
#define PLSSVM_SVM_CSVC_HPP_
#pragma once

#include "plssvm/classification_types.hpp"                 // plssvm::classification_type, plssvm::classification_type_to_full_string, plssvm::calculate_number_of_classifiers
#include "plssvm/constants.hpp"                            // plssvm::PADDING_SIZE, plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"     // plssvm::classification_data_set
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/igor_utility.hpp"                  // plssvm::detail::{has_only_named_args_v, get_value_from_named_parameter}
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::invalid_parameter_exception
#include "plssvm/gamma.hpp"                                // plssvm::calculate_gamma_value
#include "plssvm/kernel_function_types.hpp"                // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                               // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/model/classification_model.hpp"           // plssvm::classification_model
#include "plssvm/parameter.hpp"                            // plssvm::parameter
#include "plssvm/shape.hpp"                                // plssvm::shape
#include "plssvm/svm/csvm.hpp"                             // plssvm::csvm
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "igor/igor.hpp"  // igor::parser

#include <algorithm>    // std::all_of, std::merge
#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast, milliseconds}
#include <cstddef>      // std::size_t
#include <limits>       // std::numeric_limits::lowest
#include <memory>       // std::make_shared, std::dynamic_pointer_cast, std::addressof
#include <optional>     // std::make_optional
#include <tuple>        // std::tie
#include <type_traits>  // std::is_same_v
#include <utility>      // std::forward, std::move
#include <vector>       // std::vector

// forward declare svc dummy struct used in the plssvm.SVC Python bindings
struct svc;

namespace plssvm {

/**
 * @example csvc_examples.cpp
 * @brief A few examples regarding the plssvm::csvc class.
 */

/**
 * @brief Base class for all C-SVC backends.
 * @details This class implements all features shared between all C-SVC backends. It defines the whole public API of a C-SVC.
 */
class csvc : virtual public csvm {
    // befriend svc dummy struct used in the plssvm.SVC Python bindings
    friend struct ::svc;

  public:
    /// The type of the model returned by a call to the `fit` function and used in the `predict` and `score` functions.
    template <typename T>
    using model_type = ::plssvm::classification_model<T>;

    // inherit C-SVM base class constructors
    using ::plssvm::csvm::csvm;

    /**
     * @copydoc plssvm::csvm::csvm(const plssvm::csvm &)
     */
    csvc(const csvc &) = delete;
    /**
     * @copydoc plssvm::csvm::csvm(plssvm::csvm &&) noexcept
     */
    csvc(csvc &&) noexcept = default;
    /**
     * @copydoc plssvm::csvm::operator=(const plssvm::csvm &)
     */
    csvc &operator=(const csvc &) = delete;

    /**
     * @brief Correctly implement the move-assignment operator in presence of a virtual base class.
     * @details Calls the base class move-assignment operator. Afterwards, moves the potential additional `csvr` members.
     * @param[in,out] other the other C-SVM to move from
     * @return `*this`
     */
    csvc &operator=(csvc &&other) noexcept {
        if (this != std::addressof(other)) {
            ::plssvm::csvm::operator=(std::move(other));
        }
        return *this;
    }

    /**
     * @copydoc plssvm::csvm::~csvm() noexcept
     */
    ~csvc() noexcept = default;

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
     * @note For binary classification **always** one vs. all is used regardless of the provided parameter!
     * @return the learned model (`[[nodiscard]]`)
     */
    template <typename label_type, typename... Args>
    [[nodiscard]] classification_model<label_type> fit(const classification_data_set<label_type> &data, Args &&...named_args) const {
        PLSSVM_ASSERT(data.data().is_padded(), "The data points must be padded!");
        PLSSVM_ASSERT((data.data().padding() == shape{ PADDING_SIZE, PADDING_SIZE }),
                      "The provided matrix must be padded with {}, but is padded with {}!",
                      shape{ PADDING_SIZE, PADDING_SIZE },
                      data.data().padding());
#if defined(PLSSVM_ENABLE_ASSERTS)
        if (params_.kernel_type == kernel_function_type::chi_squared) {
            PLSSVM_ASSERT(std::all_of(data.data().data(), data.data().data() + data.data().size_padded(), [](const real_type val) { return val >= real_type{ 0.0 }; }),
                          "The chi-squared kernel is only well defined for non-negative values!");
        }
#endif

        if (!data.has_labels()) {
            throw invalid_parameter_exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
        }

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("fit start");

        igor::parser parser{ named_args... };

        // set default values
        // note: if the default value is changed, they must also be changed in the Python bindings!
        classification_type used_classification{ classification_type::oaa };

        // compile time check: only named parameters are permitted
        static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
        // compile time check: each named parameter must only be passed once
        static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
        // compile time check: only some named parameters are allowed
        static_assert(!parser.has_other_than(epsilon, max_iter, classification, solver), "An illegal named parameter has been passed!");

        // compile time/runtime check: the values must have the correct types
        if constexpr (parser.has(classification)) {
            // get the value of the provided named parameter
            used_classification = detail::get_value_from_named_parameter<classification_type>(parser, classification);
        }

        // start fitting the data set using a C-SVM
        const std::chrono::time_point start_time = std::chrono::steady_clock::now();

        detail::log(verbosity_level::full,
                    "Using {} ({}) as multi-class classification strategy.\n",
                    used_classification,
                    classification_type_to_full_string(used_classification));

        // copy parameter and set gamma if necessary
        parameter params{ params_ };
        // if the active gamma_type variant member isn't a real_type, replace it with a real_type value by calculating its true value based on the used data set
        // -> params.gamma is guaranteed to be a real_type now!
        params.gamma = calculate_gamma_value(params_.gamma, data.data());

        // create classification model
        classification_model<label_type> csvc_model{ params, data, used_classification };
        std::vector<unsigned long long> num_iters{};

        if (used_classification == plssvm::classification_type::oaa) {
            // use the one vs. all multi-class classification strategy
            // solve the minimization problem
            aos_matrix<real_type> alpha{};
            std::tie(alpha, *csvc_model.rho_ptr_, num_iters) = this->solve_lssvm_system_of_linear_equations(*data.data_ptr_, *data.y_ptr_, params, std::forward<Args>(named_args)...);
            csvc_model.alpha_ptr_->push_back(std::move(alpha));
        } else if (used_classification == plssvm::classification_type::oao) {
            // use the one vs. one multi-class classification strategy
            const std::size_t num_classes = data.num_classes();
            const std::size_t num_binary_classifications = calculate_number_of_classifiers(classification_type::oao, num_classes);
            const std::size_t num_features = data.num_features();
            // resize alpha_ptr_ and rho_ptr_ to the correct sizes
            csvc_model.alpha_ptr_->resize(num_binary_classifications);
            csvc_model.rho_ptr_->resize(num_binary_classifications);

            // create index vector: index_sets[0] contains the indices of all data points in the big data set with label index 0, and so on
            std::vector<std::vector<std::size_t>> index_sets(num_classes);
            {
                const std::vector<label_type> &labels = *data.labels();
                for (std::size_t i = 0; i < data.num_data_points(); ++i) {
                    index_sets[data.mapping_->get_mapped_index_by_label(labels[i])].push_back(i);
                }
            }

            if (num_classes == 2) {
                // special optimization for binary case (no temporary copies necessary)
                detail::log(verbosity_level::full,
                            "\nClassifying 0 vs 1 ({} vs {}) (1/1):\n",
                            data.mapping_->get_label_by_mapped_index(0),
                            data.mapping_->get_label_by_mapped_index(1));

                // reduce the size of the rhs (y_ptr)
                // -> consistent with the multi-class case as well as when reading the model from file in the model class constructor
                aos_matrix<real_type> reduced_y{ shape{ 1, data.y_ptr_->num_cols() } };
#pragma omp parallel for default(none) shared(data, reduced_y)
                for (std::size_t col = 0; col < data.y_ptr_->num_cols(); ++col) {
                    reduced_y(0, col) = (*data.y_ptr_)(0, col);
                }

                const auto &[alpha, rho, num_iter] = this->solve_lssvm_system_of_linear_equations(*data.data_ptr_, reduced_y, params, std::forward<Args>(named_args)...);
                csvc_model.alpha_ptr_->front() = std::move(alpha);
                csvc_model.rho_ptr_->front() = rho.front();  // prevents std::tie
                num_iters.push_back(num_iter.front());
            } else {
                // perform one vs. one classification
                std::size_t pos = 0;
                for (std::size_t i = 0; i < num_classes; ++i) {
                    for (std::size_t j = i + 1; j < num_classes; ++j) {
                        // TODO: reduce amount of copies!?
                        // assemble one vs. one classification matrix and rhs
                        const std::size_t num_data_points_in_sub_matrix{ index_sets[i].size() + index_sets[j].size() };
                        soa_matrix<real_type> binary_data{ shape{ num_data_points_in_sub_matrix, num_features }, shape{ PADDING_SIZE, PADDING_SIZE } };
                        aos_matrix<real_type> binary_y{ shape{ 1, num_data_points_in_sub_matrix } };  // note: the first dimension will always be one, since only one rhs is needed

                        // note: if this is changed, it must also be changed in the libsvm_model_parsing.hpp in the calculate_alpha_idx function!!!
                        // order the indices in increasing order
                        std::vector<std::size_t> sorted_indices(num_data_points_in_sub_matrix);
                        std::merge(index_sets[i].cbegin(), index_sets[i].cend(), index_sets[j].cbegin(), index_sets[j].cend(), sorted_indices.begin());
// copy the data points to the binary data set
#pragma omp parallel for default(none) shared(sorted_indices, binary_data, binary_y, data, index_sets) firstprivate(num_data_points_in_sub_matrix, num_features, i)
                        for (std::size_t si = 0; si < num_data_points_in_sub_matrix; ++si) {
                            for (std::size_t dim = 0; dim < num_features; ++dim) {
                                binary_data(si, dim) = (*data.data_ptr_)(sorted_indices[si], dim);
                            }
                            // needs only the check against i, since sorted_indices is guaranteed to only contain indices from i and j
                            binary_y(0, si) = detail::contains(index_sets[i], sorted_indices[si]) ? real_type{ 1.0 } : real_type{ -1.0 };
                        }

                        // solve the minimization problem -> note that only a single rhs is present
                        detail::log(verbosity_level::full,
                                    "\nClassifying {} vs {} ({} vs {}) ({}/{}):\n",
                                    i,
                                    j,
                                    data.mapping_->get_label_by_mapped_index(i),
                                    data.mapping_->get_label_by_mapped_index(j),
                                    pos + 1,
                                    calculate_number_of_classifiers(classification_type::oao, num_classes));
                        const auto &[alpha, rho, num_iter] = this->solve_lssvm_system_of_linear_equations(binary_data, binary_y, params, std::forward<Args>(named_args)...);
                        (*csvc_model.alpha_ptr_)[pos] = std::move(alpha);
                        (*csvc_model.rho_ptr_)[pos] = rho.front();  // prevents std::tie
                        num_iters.push_back(num_iter.front());
                        // go to next one vs. one classification
                        ++pos;
                        // order of the alpha value: 0 vs 1, 0 vs 2, 0 vs 3, 1 vs 2, 1 vs 3, 2 vs 3
                    }
                }
            }

            csvc_model.index_sets_ptr_ = std::make_shared<typename decltype(csvc_model.index_sets_ptr_)::element_type>(std::move(index_sets));
        }

        // move number of CG iterations to model
        csvc_model.num_iters_ = std::make_optional(std::move(num_iters));

        const std::chrono::time_point end_time = std::chrono::steady_clock::now();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "\nLearned the SVC classifier for {} multi-class classification in {}.\n\n",
                    classification_type_to_full_string(used_classification),
                    detail::tracking::tracking_entry{ "cg", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("fit end");

        return csvc_model;
    }

    //*************************************************************************************************************************************//
    //                                                          predict and score                                                          //
    //*************************************************************************************************************************************//
    /**
     * @brief Predict the labels for the @p data set using the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @param[in] data the data to predict the labels for
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the predicted labels (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const classification_model<label_type> &model, const classification_data_set<label_type> &data) const {
        PLSSVM_ASSERT(model.support_vectors().is_padded(), "The support vectors must be padded!");
        PLSSVM_ASSERT((model.support_vectors().padding() == shape{ PADDING_SIZE, PADDING_SIZE }),
                      "The support vectors must be padded with {}, but is padded with {}!",
                      shape{ PADDING_SIZE, PADDING_SIZE },
                      model.support_vectors().padding());
        PLSSVM_ASSERT(data.data().is_padded(), "The data points must be padded!");
        PLSSVM_ASSERT((data.data().padding() == shape{ PADDING_SIZE, PADDING_SIZE }),
                      "The provided predict points must be padded with {}, but is padded with {}!",
                      shape{ PADDING_SIZE, PADDING_SIZE },
                      data.data().padding());
#if defined(PLSSVM_ENABLE_ASSERTS)
        if (params_.kernel_type == kernel_function_type::chi_squared) {
            PLSSVM_ASSERT(std::all_of(data.data().data(), data.data().data() + data.data().size_padded(), [](const real_type val) { return val >= real_type{ 0.0 }; }),
                          "The chi-squared kernel is only well defined for non-negative values!");
        }
#endif

        if (model.num_features() != data.num_features()) {
            throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
        }

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("predict start");

        // convert predicted values to the correct labels
        std::vector<label_type> predicted_labels(data.num_data_points());

        PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (predict points) may never be a nullptr!");
        const soa_matrix<real_type> &predict_points = *data.data_ptr_;

        if (model.get_classification_type() == classification_type::oaa) {
            PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (model) may never be a nullptr!");
            PLSSVM_ASSERT(model.alpha_ptr_ != nullptr, "The alpha_ptr_ may never be a nullptr!");
            PLSSVM_ASSERT(model.alpha_ptr_->size() == 1, "For OAA, the alpha vector must only contain a single aos_matrix of size {}x{}!", model.num_classes(), model.num_support_vectors());
            // PLSSVM_ASSERT(model.alpha_ptr_->front().num_rows() == calculate_number_of_classifiers(classification_type::oaa, data.num_classes()), "The number of rows in the matrix must be {}, but is {}!", model.alpha_ptr_->front().num_rows(), calculate_number_of_classifiers(classification_type::oaa, data.num_classes()));

            const soa_matrix<real_type> &sv = model.support_vectors();
            const aos_matrix<real_type> &alpha = model.alpha_ptr_->front();  // num_classes x num_data_points

            // predict values using OAA -> num_data_points x num_classes
            const aos_matrix<real_type> votes = this->run_predict_values(model.params_, sv, alpha, *model.rho_ptr_, *model.w_ptr_, predict_points);

            PLSSVM_ASSERT(votes.num_rows() == data.num_data_points(), "The number of votes ({}) must be equal the number of data points ({})!", votes.num_rows(), data.num_data_points());
            PLSSVM_ASSERT(votes.num_cols() == calculate_number_of_classifiers(classification_type::oaa, model.num_classes()), "The votes contain {} values, but must contain {} values!", votes.num_cols(), calculate_number_of_classifiers(classification_type::oaa, model.num_classes()));

// use voting
#pragma omp parallel for default(none) shared(predicted_labels, votes, model) if (!std::is_same_v<label_type, bool>)
            for (std::size_t i = 0; i < predicted_labels.size(); ++i) {
                std::size_t argmax = 0;
                real_type max = std::numeric_limits<real_type>::lowest();
                for (std::size_t v = 0; v < votes.num_cols(); ++v) {
                    if (max < votes(i, v)) {
                        argmax = v;
                        max = votes(i, v);
                    }
                }
                predicted_labels[i] = std::dynamic_pointer_cast<classification_data_set<label_type>>(model.data_)->mapping_->get_label_by_mapped_index(argmax);
            }
        } else if (model.get_classification_type() == classification_type::oao) {
            PLSSVM_ASSERT(model.index_sets_ptr_ != nullptr, "The index_sets_ptr_ may never be a nullptr!");
            PLSSVM_ASSERT(model.alpha_ptr_ != nullptr, "The alpha_ptr_ may never be a nullptr!");
            PLSSVM_ASSERT(model.alpha_ptr_->size() == calculate_number_of_classifiers(classification_type::oao, model.num_classes()), "The alpha vector must contain {} matrices, but it contains {}!", calculate_number_of_classifiers(classification_type::oao, model.num_classes()), model.alpha_ptr_->size());
            PLSSVM_ASSERT(std::all_of(model.alpha_ptr_->cbegin(), model.alpha_ptr_->cend(), [](const aos_matrix<real_type> &matr) { return matr.num_rows() == 1; }), "In case of OAO, each matrix may only contain one row!");
            PLSSVM_ASSERT(model.rho_ptr_ != nullptr, "The rho_ptr_ may never be a nullptr!");
            PLSSVM_ASSERT(model.w_ptr_ != nullptr, "The w_ptr_ may never be a nullptr!");

            // predict values using OAO
            const std::size_t num_features = model.num_features();
            const std::size_t num_classes = model.num_classes();
            const std::vector<std::vector<std::size_t>> &index_sets = *model.index_sets_ptr_;

            aos_matrix<std::size_t> class_votes{ shape{ data.num_data_points(), num_classes } };

            bool calculate_w{ false };
            if (model.w_ptr_->empty()) {
                // w is currently empty
                // initialize the w matrix and calculate it later!
                calculate_w = true;
                (*model.w_ptr_) = soa_matrix<real_type>{ shape{ calculate_number_of_classifiers(classification_type::oao, num_classes), num_features }, shape{ PADDING_SIZE, PADDING_SIZE } };
            }

            // perform one vs. one prediction
            std::size_t pos = 0;
            for (std::size_t i = 0; i < num_classes; ++i) {
                for (std::size_t j = i + 1; j < num_classes; ++j) {
                    // TODO: reduce amount of copies!?
                    // assemble one vs. one classification matrix and rhs
                    const std::size_t num_data_points_in_sub_matrix{ index_sets[i].size() + index_sets[j].size() };
                    const aos_matrix<real_type> &binary_alpha = (*model.alpha_ptr_)[pos];
                    const std::vector<real_type> binary_rho{ (*model.rho_ptr_)[pos] };

                    // create binary support vector matrix, based on the number of classes
                    const soa_matrix<real_type> &binary_sv = [&]() {
                        if (num_classes == 2) {
                            // no special assembly needed in binary case
                            return model.support_vectors();
                        } else {
                            // note: if this is changed, it must also be changed in the libsvm_model_parsing.hpp in the calculate_alpha_idx function!!!
                            // order the indices in increasing order
                            soa_matrix<real_type> temp{ shape{ num_data_points_in_sub_matrix, num_features }, shape{ PADDING_SIZE, PADDING_SIZE } };
                            std::vector<std::size_t> sorted_indices(num_data_points_in_sub_matrix);
                            std::merge(index_sets[i].cbegin(), index_sets[i].cend(), index_sets[j].cbegin(), index_sets[j].cend(), sorted_indices.begin());
// copy the support vectors to the binary support vectors
// NOTE: it seems that MSVC doesn't like the collapse clause inside a lambda function
#if defined(_MSC_VER)
    #pragma omp parallel for
#else
    #pragma omp parallel for collapse(2)
#endif
                            for (std::size_t si = 0; si < num_data_points_in_sub_matrix; ++si) {
                                for (std::size_t dim = 0; dim < num_features; ++dim) {
                                    temp(si, dim) = model.support_vectors()(sorted_indices[si], dim);
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
                        soa_matrix<real_type> w{};
                        // returned w: 1 x num_features
                        binary_votes = this->run_predict_values(model.params_, binary_sv, binary_alpha, binary_rho, w, predict_points);
                        // only in case of the linear kernel, the w vector gets filled -> store it
                        if (params_.kernel_type == kernel_function_type::linear) {
#if !defined(PLSSVM_STDPAR_BACKEND_HAS_NVHPC)
    #pragma omp parallel for default(none) shared(model, w) firstprivate(num_features, pos)
#endif
                            for (std::size_t dim = 0; dim < num_features; ++dim) {
                                (*model.w_ptr_)(pos, dim) = w(0, dim);
                            }
                        }
                    } else {
                        // use previously calculated w vector
                        soa_matrix<real_type> binary_w{ shape{ 1, num_features }, shape{ PADDING_SIZE, PADDING_SIZE } };
#pragma omp parallel for default(none) shared(model, binary_w) firstprivate(num_features, pos)
                        for (std::size_t dim = 0; dim < num_features; ++dim) {
                            binary_w(0, dim) = (*model.w_ptr_)(pos, dim);
                        }
                        binary_votes = this->run_predict_values(model.params_, binary_sv, binary_alpha, binary_rho, binary_w, predict_points);
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
            for (std::size_t i = 0; i < predicted_labels.size(); ++i) {
                std::size_t argmax = 0;
                std::size_t max = 0;
                for (std::size_t v = 0; v < class_votes.num_cols(); ++v) {
                    if (max < class_votes(i, v)) {
                        argmax = v;
                        max = class_votes(i, v);
                    }
                }
                predicted_labels[i] = std::dynamic_pointer_cast<classification_data_set<label_type>>(model.data_)->mapping_->get_label_by_mapped_index(argmax);
            }
        }

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("predict end");

        return predicted_labels;
    }

    /**
     * @brief Calculate the accuracy of the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the accuracy of the model (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const classification_model<label_type> &model) const {
        return this->score(model, dynamic_cast<const classification_data_set<label_type> &>(*model.data_));
    }

    /**
     * @brief Calculate the accuracy of the labeled @p data set using the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @param[in] data the labeled data set to score
     * @throws plssvm::invalid_parameter_exception if the @p data to score has no labels
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the accuracy of the labeled @p data (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const classification_model<label_type> &model, const classification_data_set<label_type> &data) const {
        // the data set must contain labels in order to score the learned model
        if (!data.has_labels()) {
            throw invalid_parameter_exception{ "The data set to score must have labels!" };
        }
        // the number of features must be equal
        if (model.num_features() != data.num_features()) {
            throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
        }

        // predict labels
        const std::vector<label_type> predicted_labels = this->predict(model, data);
        // correct labels
        const std::vector<label_type> &correct_labels = *data.labels();

        // calculate the accuracy
        typename std::vector<label_type>::size_type correct{ 0 };
#pragma omp parallel for default(none) shared(predicted_labels, correct_labels) reduction(+ : correct)
        for (std::size_t i = 0; i < predicted_labels.size(); ++i) {
            if (predicted_labels[i] == correct_labels[i]) {
                ++correct;
            }
        }
        return static_cast<real_type>(correct) / static_cast<real_type>(predicted_labels.size());
    }
};

}  // namespace plssvm

#endif  // PLSSVM_SVM_CSVC_HPP_

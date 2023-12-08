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
#include "plssvm/constants.hpp"                   // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set.hpp"                    // plssvm::data_set
#include "plssvm/default_value.hpp"               // plssvm::default_value, plssvm::default_init
#include "plssvm/detail/igor_utility.hpp"         // plssvm::detail::{get_value_from_named_parameter, has_only_parameter_named_args_v}
#include "plssvm/detail/logging.hpp"              // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"          // plssvm::detail::memory_size
#include "plssvm/detail/operators.hpp"            // plssvm::operators::sign
#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::performance_tracker
#include "plssvm/detail/simple_any.hpp"           // plssvm::detail::simple_any
#include "plssvm/detail/type_traits.hpp"          // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"              // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                      // plssvm::aos_matrix
#include "plssvm/model.hpp"                       // plssvm::model
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/solver_types.hpp"                // plssvm::solver_type
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"            // plssvm::verbosity_level

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "igor/igor.hpp"  // igor::parser

#include <algorithm>    // std::max_element
#include <chrono>       // std::chrono::{time_point, steady_clock, duration_cast}
#include <iostream>     // std::cout, std::endl
#include <iterator>     // std::distance
#include <tuple>        // std::tie
#include <type_traits>  // std::enable_if_t, std::is_same_v, std::is_convertible_v, std::false_type
#include <utility>      // std::pair, std::forward
#include <vector>       // std::vector

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
    [[nodiscard]] target_platform get_target_platform() const noexcept { return target_; }
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
    [[nodiscard]] model<label_type> fit(const data_set<label_type> &data, Args &&...named_args) const;

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
    [[nodiscard]] std::vector<label_type> predict(const model<label_type> &model, const data_set<label_type> &data) const;

    /**
     * @brief Calculate the accuracy of the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the accuracy of the model (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const model<label_type> &model) const;
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
    [[nodiscard]] real_type score(const model<label_type> &model, const data_set<label_type> &data) const;

  protected:
    //*************************************************************************************************************************************//
    //                        pure virtual functions, must be implemented for all subclasses; doing the actual work                        //
    //*************************************************************************************************************************************//
    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @brief Calculate the total available device memory based on the used backend.
     * @return the total device memory (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual detail::memory_size get_device_memory() const = 0;
    /**
     * @brief Return the maximum allocation size possible in a single allocation.
     * @return the maximum (single) allocation size (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual detail::memory_size get_max_mem_alloc_size() const = 0;

    /**
     * @brief Setup all necessary data on the device(s). Backend specific!
     * @param[in] solver the used solver type
     * @param[in] A the data to setup
     * @return the backend specific setup data, e.g., pointer to GPU memory for the GPU related backends (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual detail::simple_any setup_data_on_devices(solver_type solver, const soa_matrix<real_type> &A) const = 0;

    /**
     * @brief Explicitly assemble the kernel matrix. Backend specific!
     * @param[in] solver the used solver type, determines the return type
     * @param[in] params the parameters used to assemble the kernel matrix (e.g., the used kernel function)
     * @param[in] data the data used to assemble the kernel matrix; fully stored on the device
     * @param[in] q_red the vector used in the dimensional reduction
     * @param[in] QA_cost the value used in the dimensional reduction
     * @return based on the used solver type (e.g., cg_explicit -> kernel matrix fully stored on the device; cg_implicit -> "nothing") (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual detail::simple_any assemble_kernel_matrix(solver_type solver, const parameter &params, const ::plssvm::detail::simple_any &data, const std::vector<real_type> &q_red, real_type QA_cost) const = 0;

    /**
     * @brief Perform a BLAS level 3 matrix-matrix multiplication: `C = alpha * A * B + beta * C`.
     * @param[in] solver the used solver type, determines the type of @p A
     * @param[in] alpha the value to scale the result of the matrix-matrix multiplication
     * @param[in] A a matrix depending on the used solver type (e.g., cg_explicit -> the kernel matrix fully stored on the device; cg_implicit -> the input data set used to implicitly perfom the matrix-matrix multiplication)
     * @param[in] B the other matrix to multiply the kernel matrix with
     * @param[in] beta the value to scale the matrix o add with
     * @param[in,out] C the result matrix and the matrix to add (inplace)
     */
    virtual void blas_level_3(solver_type solver, real_type alpha, const detail::simple_any &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const = 0;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] params the SVM parameters used in the respective kernel functions
     * @param[in] support_vectors the previously learned support vectors
     * @param[in] alpha the alpha values (weights) associated with the support vectors and classes
     * @param[in] rho the rho values for each class determined after training the model
     * @param[in,out] w the normal vectors to speedup prediction in case of the linear kernel function, an empty vector in case of the polynomial or rbf kernel
     * @param[in] predict_points the points to predict
     * @throws plssvm::exception any exception thrown by the backend's implementation
     * @return a vector filled with the predictions (not the actual labels!) (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual aos_matrix<real_type> predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const = 0;

    /// The target platform of this SVM.
    target_platform target_{ plssvm::target_platform::automatic };

  protected:  // necessary for tests, would otherwise be private
    /**
     * @brief Perform some sanity checks on the passed SVM parameters.
     * @throws plssvm::invalid_parameter_exception if the kernel function is invalid
     * @throws plssvm::invalid_parameter_exception if the gamma value for the polynomial or radial basis function kernel is **not** greater than zero
     */
    void sanity_check_parameter() const;

    /**
     * @brief Solve the system of linear equations `K * X = B` where `K` is the kernel matrix assembled from @p A using the @p params with potentially multiple right-hand sides.
     * @tparam Args the type of the potential additional parameters
     * @param[in] A the data used to create the kernel matrix
     * @param[in] B the right-hand sides
     * @param[in] params the parameter to create the kernel matrix
     * @param[in] named_args additional parameters for the respective algorithm used to solve the system of linear equations
     * @return the result matrix `X`, the respective biases, and the number of iterations necessary to solve the system of linear equations (`[[nodiscard]]`)
     */
    template <typename... Args>
    [[nodiscard]] std::tuple<aos_matrix<real_type>, std::vector<real_type>, unsigned long long> solve_lssvm_system_of_linear_equations(const soa_matrix<real_type> &A, const aos_matrix<real_type> &B, const parameter &params, Args &&...named_args) const;
    /**
     * @brief Solve the system of linear equations `AX = B` where `A` is the kernel matrix using the Conjugate Gradients (CG) algorithm.
     * @param[in] A the kernel matrix
     * @param[in] B the right-hand sides
     * @param[in] eps the termination criterion for the CG algorithm
     * @param[in] max_cg_iter the maximum number of CG iterations
     * @param[in] cd_solver_variant the variation of the CG algorithm to use, i.e., how the kernel matrix is assembled (currently: explicit, streaming, implicit)
     * @return the result matrix `X` and the number of CG iterations necessary to solve the system of linear equations (`[[nodiscard]]`)
     */
    [[nodiscard]] std::pair<soa_matrix<real_type>, unsigned long long> conjugate_gradients(const detail::simple_any &A, const soa_matrix<real_type> &B, real_type eps, unsigned long long max_cg_iter, solver_type cd_solver_variant) const;
    /**
     * @brief Perform a dimensional reduction for the kernel matrix.
     * @details Reduces the resulting dimension by `2` compared to the original LS-SVM formulation.
     * @param[in] params the parameter used for the kernel matrix
     * @param[in] A the data used for the kernel matrix
     * @return the reduction vector ´q_red` and the bottom-right value `QA_cost` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::pair<std::vector<real_type>, real_type> perform_dimensional_reduction(const parameter &params, const soa_matrix<real_type> &A) const;

    /**
     * @copydoc plssvm::csvm::blas_level_3
     * @detail Small wrapper around the virtual `plssvm::csvm::blas_level_3` function to easily track its execution time.
     */
    [[nodiscard]] std::chrono::duration<long, std::milli> run_blas_level_3(solver_type cg_solver, real_type alpha, const detail::simple_any &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const;

    /**
     * @copydoc plssvm::csvm::predict_values
     * @detail Small wrapper around the virtual `plssvm::csvm::predict_values` function to easily track its execution time.
     */
    [[nodiscard]] aos_matrix<real_type> run_predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const;

  private:
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

template <typename label_type, typename... Args>
model<label_type> csvm::fit(const data_set<label_type> &data, Args &&...named_args) const {
    PLSSVM_ASSERT(data.data().is_padded(), "The data points must be padded!");
    PLSSVM_ASSERT(data.data().padding()[0] == PADDING_SIZE && data.data().padding()[1] == PADDING_SIZE,
                  "The provided matrix must be padded with [{}, {}], but is padded with [{}, {}]!",
                  PADDING_SIZE,
                  PADDING_SIZE,
                  data.data().padding()[0],
                  data.data().padding()[1]);

    if (!data.has_labels()) {
        throw invalid_parameter_exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
    }

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
    if (params.gamma.is_default()) {
        // no gamma provided -> use default value which depends on the number of features in the data set
        params.gamma = real_type{ 1.0 } / data.num_features();
    }

    // create model
    model<label_type> csvm_model{ params, data, used_classification };
    std::vector<unsigned long long> num_iters{};

    if (used_classification == plssvm::classification_type::oaa) {
        // use the one vs. all multi-class classification strategy
        // solve the minimization problem
        aos_matrix<real_type> alpha;
        unsigned long long num_iter{};
        std::tie(alpha, *csvm_model.rho_ptr_, num_iter) = solve_lssvm_system_of_linear_equations(*data.data_ptr_, *data.y_ptr_, params, std::forward<Args>(named_args)...);
        csvm_model.alpha_ptr_->push_back(std::move(alpha));
        num_iters.resize(calculate_number_of_classifiers(used_classification, data.num_classes()), num_iter);
    } else if (used_classification == plssvm::classification_type::oao) {
        // use the one vs. one multi-class classification strategy
        const std::size_t num_classes = data.num_classes();
        const std::size_t num_binary_classifications = calculate_number_of_classifiers(classification_type::oao, num_classes);
        const std::size_t num_features = data.num_features();
        // resize alpha_ptr_ and rho_ptr_ to the correct sizes
        csvm_model.alpha_ptr_->resize(num_binary_classifications);
        csvm_model.rho_ptr_->resize(num_binary_classifications);

        // create index vector: index_sets[0] contains the indices of all data points in the big data set with label index 0, and so on
        std::vector<std::vector<std::size_t>> index_sets(num_classes);
        {
            const std::vector<label_type> &labels = data.labels().value();
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
            aos_matrix<real_type> reduced_y{ 1, data.y_ptr_->num_cols() };
            #pragma omp parallel for default(none) shared(data, reduced_y)
            for (std::size_t col = 0; col < data.y_ptr_->num_cols(); ++col) {
                reduced_y(0, col) = (*data.y_ptr_)(0, col);
            }

            const auto &[alpha, rho, num_iter] = solve_lssvm_system_of_linear_equations(*data.data_ptr_, reduced_y, params, std::forward<Args>(named_args)...);
            csvm_model.alpha_ptr_->front() = std::move(alpha);
            csvm_model.rho_ptr_->front() = rho.front();  // prevents std::tie
            num_iters.push_back(num_iter);
        } else {
            // perform one vs. one classification
            std::size_t pos = 0;
            for (std::size_t i = 0; i < num_classes; ++i) {
                for (std::size_t j = i + 1; j < num_classes; ++j) {
                    // TODO: reduce amount of copies!?
                    // assemble one vs. one classification matrix and rhs
                    const std::size_t num_data_points_in_sub_matrix{ index_sets[i].size() + index_sets[j].size() };
                    soa_matrix<real_type> binary_data{ num_data_points_in_sub_matrix, num_features, PADDING_SIZE, PADDING_SIZE };
                    aos_matrix<real_type> binary_y{ 1, num_data_points_in_sub_matrix };  // note: the first dimension will always be one, since only one rhs is needed

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
                    const auto &[alpha, rho, num_iter] = solve_lssvm_system_of_linear_equations(binary_data, binary_y, params, std::forward<Args>(named_args)...);
                    (*csvm_model.alpha_ptr_)[pos] = std::move(alpha);
                    (*csvm_model.rho_ptr_)[pos] = rho.front();  // prevents std::tie
                    num_iters.push_back(num_iter);
                    // go to next one vs. one classification
                    ++pos;
                    // order of the alpha value: 0 vs 1, 0 vs 2, 0 vs 3, 1 vs 2, 1 vs 3, 2 vs 3
                }
            }
        }

        csvm_model.index_sets_ptr_ = std::make_shared<typename decltype(csvm_model.index_sets_ptr_)::element_type>(std::move(index_sets));
    }

    // move number of CG iterations to model
    csvm_model.num_iters_ = std::make_optional(std::move(num_iters));

    const std::chrono::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "\nLearned the SVM classifier for {} multi-class classification in {}.\n\n",
                classification_type_to_full_string(used_classification),
                detail::tracking_entry{ "cg", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    return csvm_model;
}

template <typename label_type>
std::vector<label_type> csvm::predict(const model<label_type> &model, const data_set<label_type> &data) const {
    PLSSVM_ASSERT(model.support_vectors().is_padded(), "The support vectors must be padded!");
    PLSSVM_ASSERT(model.support_vectors().padding()[0] == PADDING_SIZE && model.support_vectors().padding()[1] == PADDING_SIZE,
                  "The support vectors must be padded with [{}, {}], but is padded with [{}, {}]!",
                  PADDING_SIZE,
                  PADDING_SIZE,
                  model.support_vectors().padding()[0],
                  model.support_vectors().padding()[1]);
    PLSSVM_ASSERT(data.data().is_padded(), "The data points must be padded!");
    PLSSVM_ASSERT(data.data().padding()[0] == PADDING_SIZE && data.data().padding()[1] == PADDING_SIZE,
                  "The provided predict points must be padded with [{}, {}], but is padded with [{}, {}]!",
                  PADDING_SIZE,
                  PADDING_SIZE,
                  data.data().padding()[0],
                  data.data().padding()[1]);


    if (model.num_features() != data.num_features()) {
        throw invalid_parameter_exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", data.num_features(), model.num_features()) };
    }

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(data.num_data_points());

    PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (predict points) may never be a nullptr!");
    const soa_matrix<real_type> &predict_points = *data.data_ptr_;

    if (model.get_classification_type() == classification_type::oaa) {
        PLSSVM_ASSERT(data.data_ptr_ != nullptr, "The data_ptr_ (model) may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_ != nullptr, "The alpha_ptr_ may never be a nullptr!");
        PLSSVM_ASSERT(model.alpha_ptr_->size() == 1, "For OAA, the alpha vector must only contain a single aos_matrix of size {}x{}!", model.num_classes(), model.num_support_vectors());
        PLSSVM_ASSERT(model.alpha_ptr_->front().num_rows() == calculate_number_of_classifiers(classification_type::oaa, data.num_classes()), "The number of rows in the matrix must be {}, but is {}!", model.alpha_ptr_->front().num_rows(), calculate_number_of_classifiers(classification_type::oaa, data.num_classes()));
        PLSSVM_ASSERT(model.alpha_ptr_->front().num_cols() == data.num_data_points(), "The number of weights ({}) must be equal to the number of support vectors ({})!", model.alpha_ptr_->front().num_cols(), data.num_data_points());

        const soa_matrix<real_type> &sv = *model.data_.data_ptr_;
        const aos_matrix<real_type> &alpha = model.alpha_ptr_->front();  // num_classes x num_data_points

        // predict values using OAA -> num_data_points x num_classes
        const aos_matrix<real_type> votes = this->run_predict_values(model.params_, sv, alpha, *model.rho_ptr_, *model.w_ptr_, predict_points);

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

        aos_matrix<std::size_t> class_votes{ data.num_data_points(), num_classes };

        bool calculate_w{ false };
        if (model.w_ptr_->empty()) {
            // w is currently empty
            // initialize the w matrix and calculate it later!
            calculate_w = true;
            (*model.w_ptr_) = soa_matrix<real_type>{ calculate_number_of_classifiers(classification_type::oao, num_classes), num_features };
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
                        return *model.data_.data_ptr_;
                    } else {
                        // note: if this is changed, it must also be changed in the libsvm_model_parsing.hpp in the calculate_alpha_idx function!!!
                        // order the indices in increasing order
                        soa_matrix<real_type> temp{ num_data_points_in_sub_matrix, num_features, PADDING_SIZE, PADDING_SIZE };
                        std::vector<std::size_t> sorted_indices(num_data_points_in_sub_matrix);
                        std::merge(index_sets[i].cbegin(), index_sets[i].cend(), index_sets[j].cbegin(), index_sets[j].cend(), sorted_indices.begin());
                        // copy the support vectors to the binary support vectors
                        #pragma omp parallel for collapse(2)
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
                    soa_matrix<real_type> w{};
                    // returned w: 1 x num_features
                    binary_votes = this->run_predict_values(model.params_, binary_sv, binary_alpha, binary_rho, w, predict_points);
                    // only in case of the linear kernel, the w vector gets filled -> store it
                    if (params_.kernel_type == kernel_function_type::linear) {
                        #pragma omp parallel for default(none) shared(model, w) firstprivate(num_features, pos)
                        for (std::size_t dim = 0; dim < num_features; ++dim) {
                            (*model.w_ptr_)(pos, dim) = w(0, dim);
                        }
                    }
                } else {
                    // use previously calculated w vector
                    soa_matrix<real_type> binary_w{ 1, num_features };
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
        for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
            std::size_t argmax = 0;
            real_type max = std::numeric_limits<real_type>::lowest();
            for (std::size_t v = 0; v < class_votes.num_cols(); ++v) {
                if (max < class_votes(i, v)) {
                    argmax = v;
                    max = static_cast<real_type>(class_votes(i, v));
                }
            }
            predicted_labels[i] = model.data_.mapping_->get_label_by_mapped_index(argmax);
        }
    }

    return predicted_labels;
}

template <typename label_type>
real_type csvm::score(const model<label_type> &model) const {
    return this->score(model, model.data_);
}

template <typename label_type>
real_type csvm::score(const model<label_type> &model, const data_set<label_type> &data) const {
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

//*************************************************************************************************************************************//
//                                                       private member functions                                                      //
//*************************************************************************************************************************************//

template <typename... Args>
std::tuple<aos_matrix<real_type>, std::vector<real_type>, unsigned long long> csvm::solve_lssvm_system_of_linear_equations(const soa_matrix<real_type> &A, const aos_matrix<real_type> &B, const parameter &params, Args &&...named_args) const {
    PLSSVM_ASSERT(!A.empty(), "The A matrix may not be empty!");
    PLSSVM_ASSERT(A.is_padded(), "The A matrix must be padded!");
    PLSSVM_ASSERT(A.padding()[0] == PADDING_SIZE && A.padding()[1] == PADDING_SIZE,
                  "The provided matrix must be padded with [{}, {}], but is padded with [{}, {}]!",
                  PADDING_SIZE,
                  PADDING_SIZE,
                  A.padding()[0],
                  A.padding()[1]);
    PLSSVM_ASSERT(!B.empty(), "The B matrix may not be empty!");
    PLSSVM_ASSERT(A.num_rows() == B.num_cols(), "The number of data points in A ({}) and B ({}) must be the same!", A.num_rows(), B.num_cols());

    igor::parser parser{ std::forward<Args>(named_args)... };

    // set default values
    // note: if the default values are changed, they must also be changed in the Python bindings!
    auto used_epsilon{ plssvm::real_type{ 0.001 } };
    unsigned long long used_max_iter{ A.num_rows() - 1 };  // account for later dimensional reduction
    solver_type used_solver{ solver_type::automatic };

    // compile time check: only named parameters are permitted
    static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!parser.has_other_than(epsilon, max_iter, classification, solver), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (parser.has(epsilon)) {
        // get the value of the provided named parameter
        used_epsilon = detail::get_value_from_named_parameter<real_type>(parser, epsilon);
        // check if value makes sense
        if (used_epsilon <= real_type{ 0.0 }) {
            throw invalid_parameter_exception{ fmt::format("epsilon must be less than 0.0, but is {}!", used_epsilon) };
        }
    }
    if constexpr (parser.has(max_iter)) {
        // get the value of the provided named parameter
        used_max_iter = detail::get_value_from_named_parameter<unsigned long long>(parser, max_iter);
        // check if value makes sense
        if (used_max_iter == 0) {
            throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", used_max_iter) };
        }
    }
    if constexpr (parser.has(solver)) {
        // get the value of the provided parameter
        used_solver = detail::get_value_from_named_parameter<solver_type>(parser, solver);
    }

    const std::size_t num_rows = A.num_rows();
    const std::size_t num_features = A.num_cols();
    const std::size_t num_rows_reduced = num_rows - 1;
    const std::size_t num_rhs = B.num_rows();

    // determine the correct solver type, if the automatic solver type has been provided
    if (used_solver == solver_type::automatic) {
        using namespace detail::literals;

        // define used safety margin constants
        constexpr detail::memory_size minimal_safety_margin = 512_MiB;
        constexpr long double percentual_safety_margin = 0.05L;
        const auto reduce_total_memory = [=](const detail::memory_size total_memory) {
            return total_memory - std::max(total_memory * percentual_safety_margin, minimal_safety_margin);
        };

        const detail::memory_size total_system_memory = detail::get_system_memory();
        const detail::memory_size total_device_memory = this->get_device_memory();

        // 4B/8B * (data_set size including padding + explicit kernel matrix size + B and C matrix in GEMM + q_red vector)
#if defined(PLSSVM_USE_GEMM)
        const unsigned long long kernel_matrix_size = (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE);
#else
        const unsigned long long kernel_matrix_size = (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2;
#endif
        const detail::memory_size total_memory_needed{ sizeof(real_type) * ((num_rows + PADDING_SIZE) * (num_features + PADDING_SIZE) + kernel_matrix_size + 2 * num_rows_reduced * num_rhs + num_features) };

        detail::log(verbosity_level::full,
                    "Determining the solver type based on the available memory:\n"
                    "  - total system memory: {}\n"
                    "  - usable system memory: {} = {}\n"
                    "  - total device memory: {}\n"
                    "  - usable device memory: {} = {}\n"
                    "  - memory needed: {}\n",
                    detail::tracking_entry{ "solver", "system_memory", total_system_memory },
                    fmt::format("{} {}", total_system_memory, total_system_memory * percentual_safety_margin > minimal_safety_margin ? "* 0.95" : "- 512 MiB"),
                    detail::tracking_entry{ "solver", "available_system_memory", reduce_total_memory(total_system_memory) },
                    detail::tracking_entry{ "solver", "device_memory", total_device_memory },
                    fmt::format("{} {}", total_device_memory, total_device_memory * percentual_safety_margin > minimal_safety_margin ? "* 0.95" : "- 512 MiB"),
                    detail::tracking_entry{ "solver", "available_device_memory", reduce_total_memory(total_device_memory) },
                    detail::tracking_entry{ "solver", "needed_memory", total_memory_needed });

        // select solver type based on the available memory
        if (total_memory_needed < total_device_memory) {
            used_solver = solver_type::cg_explicit;
        } else if (total_memory_needed > total_device_memory && total_memory_needed < total_system_memory) {
            used_solver = solver_type::cg_streaming;
        } else {
            used_solver = solver_type::cg_implicit;
        }

#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        // enforce max mem alloc size if requested
        const detail::memory_size max_mem_alloc_size = this->get_max_mem_alloc_size();
        // maximum of data set and kernel matrix
        const detail::memory_size max_single_allocation_size{ sizeof(real_type) * std::max<unsigned long long>(num_rows * num_features, kernel_matrix_size) };
        detail::log(verbosity_level::full,
                    "  - max. memory allocation size: {}\n",
                    detail::tracking_entry{ "solver", "device_max_mem_alloc_size", max_mem_alloc_size });

        // note: only cg_explicit currently implemented
        // TODO: also implement logic for cg_streaming and cg_implicit
        if (used_solver == solver_type::cg_explicit && max_single_allocation_size > max_mem_alloc_size) {
            detail::log(verbosity_level::full,
                        "The biggest single allocation ({}) exceeds the guaranteed maximum memory allocation size ({}), falling back to solver_type::cg_streaming.\n",
                        max_single_allocation_size,
                        max_mem_alloc_size);
            plssvm::detail::log(verbosity_level::full | verbosity_level::warning,
                                "WARNING: if you are sure that the guaranteed maximum memory allocation size can be safely ignored on your device, "
                                "this check can be disabled via \"-DPLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE=OFF\" during the CMake configuration!\n");
            used_solver = solver_type::cg_streaming;
        }
#endif
    }

    detail::log(verbosity_level::full,
                "Using {} as solver for AX=B.\n\n",
                detail::tracking_entry{ "solver", "solver_type", used_solver });

    // perform dimensional reduction
    // note: structured binding is rejected by clang HIP compiler!
    std::vector<real_type> q_red{};
    real_type QA_cost{};
    std::tie(q_red, QA_cost) = this->perform_dimensional_reduction(params, A);

    // update right-hand sides (B)
    std::vector<real_type> b_back_value(num_rhs);
    soa_matrix<real_type> B_red{ num_rhs, num_rows_reduced };
    #pragma omp parallel for default(none) shared(B, B_red, b_back_value) firstprivate(num_rhs, num_rows_reduced)
    for (std::size_t row = 0; row < num_rhs; ++row) {
        b_back_value[row] = B(row, num_rows_reduced);
        for (std::size_t col = 0; col < num_rows_reduced; ++col) {
            B_red(row, col) = B(row, col) - b_back_value[row];
        }
    }

    // setup/allocate necessary data on the device(s)
    const detail::simple_any data = this->setup_data_on_devices(used_solver, A);

    // assemble explicit kernel matrix
    const std::chrono::steady_clock::time_point assembly_start_time = std::chrono::steady_clock::now();
    const detail::simple_any kernel_matrix = this->assemble_kernel_matrix(used_solver, params, data, q_red, QA_cost);
    const std::chrono::steady_clock::time_point assembly_end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Assembled the kernel matrix in {}.\n",
                detail::tracking_entry{ "kernel_matrix", "kernel_matrix_assembly", std::chrono::duration_cast<std::chrono::milliseconds>(assembly_end_time - assembly_start_time) });

    // choose the correct algorithm based on the (provided) solver type -> currently only CG available
    soa_matrix<real_type> X;
    unsigned long long num_iter{};
    std::tie(X, num_iter) = conjugate_gradients(kernel_matrix, B_red, used_epsilon, used_max_iter, used_solver);  // TODO: q_red for implicit

    // calculate bias and undo dimensional reduction
    aos_matrix<real_type> X_ret{ num_rhs, A.num_rows(), PADDING_SIZE, PADDING_SIZE };
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

    return std::make_tuple(std::move(X_ret), std::move(bias), num_iter);
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
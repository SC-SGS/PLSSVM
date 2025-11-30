/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/classification_types.hpp"              // plssvm::classification_type
#include "plssvm/constants.hpp"                         // plssvm::real_type, plssvm::DEFAULT_EPSILON
#include "plssvm/csvm_factory.hpp"                      // plssvm::make_csvc
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/detail/assert.hpp"                     // PLSSVM_ASSERT
#include "plssvm/detail/type_traits.hpp"                // plssvm::detail::remove_cvref_t
#include "plssvm/gamma.hpp"                             // plssvm::gamma_coefficient_type, plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"             // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                            // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/model/classification_model.hpp"        // plssvm::classification_model
#include "plssvm/parameter.hpp"                         // plssvm::parameter, named arguments definition
#include "plssvm/svm/csvc.hpp"                          // plssvm::csvc
#include "plssvm/verbosity_levels.hpp"                  // plssvm::verbosity_level, plssvm::verbosity

#include "bindings/Python/bindings_fwd.hpp"                             // forward declare all helper functions to create the Python bindings
#include "bindings/Python/data_set/variant_wrapper.hpp"                 // plssvm::bindings::python::util::classification_data_set_wrapper
#include "bindings/Python/model/variant_wrapper.hpp"                    // plssvm::bindings::python::util::classification_model_wrapper
#include "bindings/Python/type_caster/label_vector_wrapper_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::bindings::python::label_vector_wrapper
#include "bindings/Python/type_caster/matrix_type_caster.hpp"           // NOLINT: a custom Pybind11 type caster for a plssvm::matrix
#include "bindings/Python/type_caster/matrix_wrapper_type_caster.hpp"   // a custom Pybind11 type caster for a plssvm::bindings::python::util::matrix_wrapper
#include "bindings/Python/utility.hpp"                                  // plssvm::bindings::python::util::{check_kwargs_for_correctness, vector_to_pyarray}

#include "fmt/format.h"            // fmt::format
#include "fmt/ranges.h"            // fmt::join
#include "pybind11/buffer_info.h"  // py::buffer_info
#include "pybind11/cast.h"         // py::cast, py::arg
#include "pybind11/numpy.h"        // support for STL types
#include "pybind11/operators.h"    // NOLINT: support for operators
#include "pybind11/pybind11.h"     // py::module_, py::class_, py::init, py::return_value_policy, py::self, py::dynamic_attr, py::value_error, py::attribute_error, py::tuple, py::pickle
#include "pybind11/pytypes.h"      // py::dict, py::kwargs, py::str
#include "pybind11/stl.h"          // NOLINT: support for STL types

#include <algorithm>  // std::fill
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int32_t
#include <exception>  // std::exception
#include <map>        // std::map
#include <memory>     // std::unique_ptr, std::make_unique
#include <optional>   // std::optional, std::nullopt
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <tuple>      // std::make_tuple, std::ignore
#include <utility>    // std::move
#include <variant>    // std::holds_alternative, std::variant, std::visit
#include <vector>     // std::vector

namespace py = pybind11;

// TODO: implement missing functionality (as far es possible)
/*
 * Currently missing:
 * - shrinking constructor parameter (makes no sense for LS-SVMs)
 * - probability constructor parameter (needs Platt scaling -> complex)
 * - cache_size constructor parameter (not applicable in PLSSVM)
 * - class_weight constructor parameter
 * - break_ties constructor parameter
 * - random_state constructor parameter (needed for probability estimates)
 * - dual_coef_ attribute
 * - probA_ attribute (needed for probability estimates)
 * - probB_ attribute (needed for probability estimates)
 * - get_metadata_routing function (no idea how to implement this function)
 * - predict_log_proba function (needed for probability estimates)
 * - predict_proba function (needed for probability estimates)
 * - set_fit_request function (no idea how to implement this function)
 * - set_score_request function (no idea how to implement this function)
 * - sample_weight parameter for the fit function
 * - sample_weight parameter for the score function
 */

// dummy
struct svc {
    using possible_vector_types = typename plssvm::bindings::python::util::classification_data_set_wrapper::possible_vector_types;
    using possible_data_set_types = typename plssvm::bindings::python::util::classification_data_set_wrapper::possible_data_set_types;
    using possible_model_types = typename plssvm::bindings::python::util::classification_model_wrapper::possible_model_types;

    /**
     * @brief Construct a default svc wrapper doing nothing.
     */
    svc() :
        svm_{ plssvm::make_csvc(plssvm::gamma = plssvm::gamma_coefficient_type::scale) } { }

    /**
     * @brief Construct a new svc wrapper with the provided parameters.
     * @param[in] params the SVM hyper-parameters
     * @param[in] epsilon the epsilon value for the CG termination criterion
     * @param[in] max_iter the maximum number of CG iterations
     * @param[in] classification the classfication type (or decision function shape)
     */
    svc(const plssvm::parameter params, const plssvm::real_type epsilon, const std::optional<unsigned long long> max_iter, const plssvm::classification_type classification) :
        svm_{ plssvm::make_csvc(params) },
        epsilon_{ epsilon },
        max_iter_{ max_iter },
        classification_{ classification } { }

    /**
     * @brief Wrapper function to call the private (friendship) predict_values function.
     * @tparam Args the types of the parameter used for calling the predict_values function
     * @param[in] args the predict_values function parameter
     * @return the predicted values (`[[nodiscard]]`)
     */
    template <typename... Args>
    [[nodiscard]] auto call_predict_values(Args &&...args) const {
        PLSSVM_ASSERT(svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
        return svm_->predict_values(std::forward<Args>(args)...);
    }

    /**
     * @brief Get the index sets used for the decision_function function in the one-vs-one classification case from the currently learned model.
     * @return the index sets (`[[nodiscard]]`)
     */
    [[nodiscard]] const auto &get_index_sets_ptr() const {
        if (model_ == nullptr) {
            throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
        }

        // clang-format off
        return std::visit([](auto &&model) -> const auto & {
            return *model.index_sets_ptr_;
        }, *model_);
        // clang-format on
    }

    /**
     * @brief Get the w values used for the coef_ attribute from the currently learned linear model.
     * @return the w values (`[[nodiscard]]`)
     */
    [[nodiscard]] const auto &get_w_ptr() const {
        if (model_ == nullptr) {
            throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
        }

        // clang-format off
        return std::visit([](auto &&model) -> const auto & {
            return *model.w_ptr_;
        }, *model_);
        // clang-format on
    }

    /**
     * @brief Return the currently used params.
     * @details Necessary for the same Python function and also the string representation.
     * @params[in] deep_copy *unused*
     * @return a Python dictionary containing the used parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] py::dict get_params([[maybe_unused]] const bool deep_copy) const {
        PLSSVM_ASSERT(svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
        const plssvm::parameter params = svm_->get_params();

        // fill a Python dictionary with the supported keys and values
        py::dict py_params;
        py_params["C"] = params.cost;
        // py_params["break_ties"] = false;
        // py_params["cache_size"] = 0;
        // py_params["class_weight"] = py::none{};
        py_params["coef0"] = params.coef0;
        py_params["decision_function_shape"] = classification_ == plssvm::classification_type::oaa ? "ovr" : "ovo";
        py_params["degree"] = params.degree;
        if (std::holds_alternative<plssvm::real_type>(params.gamma)) {
            py_params["gamma"] = std::get<plssvm::real_type>(params.gamma);
        } else {
            // can't use this for both or the numeric value would also be interpreted as a string like '0.001'
            py_params["gamma"] = fmt::format("{}", params.gamma);
        }
        py_params["kernel"] = fmt::format("{}", params.kernel_type);
        py_params["max_iter"] = max_iter_.has_value() ? static_cast<long long>(max_iter_.value()) : -1;
        // py_params["probability"] = false;
        // py_params["random_state"] = py::none{};
        // py_params["shrinking"] = false;
        py_params["tol"] = epsilon_;
        py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;

        return py_params;
    }

    /**
     * @brief Calculate the support vector indices per class.
     * @return the support vector indices (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<int> calculate_sv_indices_per_class() const {
        PLSSVM_ASSERT(model_ != nullptr, "model_ may not be a nullptr! Maybe you forgot to initialize it?");

        return std::visit([&](auto &&model) {
            using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;

            std::map<label_type, std::vector<int>> indices_per_class{};
            // init index-map map
            for (const label_type &label : model.classes()) {
                indices_per_class.insert({ label, std::vector<int>{} });
            }
            // sort the indices into the respective bucket based on their associated class
            for (std::size_t idx = 0; idx < model.num_support_vectors(); ++idx) {
                indices_per_class[model.labels()->get()[idx]].push_back(static_cast<int>(idx));
            }
            // convert map values to vector
            std::vector<int> support{};
            support.reserve(model.num_support_vectors());
            for (const auto &[label, indices] : indices_per_class) {
                support.insert(support.cend(), indices.cbegin(), indices.cend());
            }
            return support;
        },
                          *model_);
    }

    /// Pointer to the the stored PLSSVM C-SVC instance.
    std::unique_ptr<plssvm::csvc> svm_;
    /// The CG termination criterion if provided.
    plssvm::real_type epsilon_{};
    /// The maximum number of CG iterations if provided.
    std::optional<unsigned long long> max_iter_;
    /// The used classification type (or decision function shape).
    plssvm::classification_type classification_{};

    /// The data type of the labels.
    py::dtype py_dtype_;
    /// Pointer to the classification data set wrapper (represents data sets with all possible label types).
    std::unique_ptr<possible_data_set_types> data_;
    /// Pointer to the classification model wrapper (represents models with all possible label types).
    std::unique_ptr<possible_model_types> model_;

    /// The name of the features. Can only be provided via a Pandas DataFrame.
    std::optional<std::vector<std::string>> feature_names_;
};

void init_sklearn_svc(py::module_ &m) {
    // documentation based on sklearn.svm.SVC documentation
    py::class_<svc> py_svc(m, "SVC", py::dynamic_attr(), "A C-SVC implementation adhering to sklearn.svm.SVC using PLSSVM as backend.");
    py_svc.def(py::init([](const plssvm::real_type C, const plssvm::kernel_function_type kernel, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0, const plssvm::real_type tol, const bool verbose, const long long max_iter, const plssvm::classification_type decision_function_shape) {
                   // sanity check parameters
                   if (max_iter < -1) {
                       throw py::value_error{ fmt::format("max_iter must either be greater than zero or -1, got {}!", max_iter) };
                   }

                   // set verbosity
                   if (verbose) {
                       if (plssvm::verbosity == plssvm::verbosity_level::quiet) {
                           // if current verbosity is quiet, override with full verbosity, since 'verbose=TRUE' should never result in no output
                           plssvm::verbosity = plssvm::verbosity_level::full;
                       }
                       // otherwise: use currently active verbosity level
                   } else {
                       plssvm::verbosity = plssvm::verbosity_level::quiet;
                   }

                   // create plssvm::parameter struct
                   const plssvm::parameter params{ kernel, degree, gamma, coef0, C };
                   // we use an unsigned type for max_iter -> convert it to an optional to support -1
                   const std::optional<unsigned long long> used_max_iter = max_iter == -1 ? std::nullopt : std::make_optional(static_cast<unsigned long long>(max_iter));
                   // create SVC wrapper
                   return svc{ params, tol, used_max_iter, decision_function_shape };
               }),
               "Construct a new SVC classifier.",
               py::kw_only(),
               py::arg("C") = 1.0,
               py::arg("kernel") = plssvm::kernel_function_type::rbf,
               py::arg("degree") = 3,
               py::arg("gamma") = plssvm::gamma_coefficient_type::scale,
               py::arg("coef0") = 0.0,
               // py::arg("shrinking") = true,
               // py::arg("probability") = false,
               py::arg("tol") = plssvm::DEFAULT_EPSILON,
               // py::arg("cache_size") = 200,
               // py::arg("class_weight") = py::none{},
               py::arg("verbose") = false,
               py::arg("max_iter") = -1,
               py::arg("decision_function_shape") = plssvm::classification_type::oaa
               // py::arg("break_ties") = false,
               // py::arg("random_state") = py::none{}
    );

    //*************************************************************************************************************************************//
    //                                                             ATTRIBUTES                                                              //
    //*************************************************************************************************************************************//
    py_svc
        .def_property_readonly("class_weight_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'class_weight_'" };
            }

            // note: constant zero since the class_weight parameter is currently not supported
            const auto size = static_cast<int>(std::visit([](auto &&model) { return model.num_classes(); }, *self.model_));
            py::array_t<plssvm::real_type, py::array::c_style> py_array(size);
            const py::buffer_info buffer = py_array.request();
            auto ptr = static_cast<plssvm::real_type *>(buffer.ptr);
            std::fill(ptr, ptr + size, plssvm::real_type{ 1.0 });
            return py_array; }, "Multipliers of parameter C for each class. ndarray of shape (n_classes,)")
        .def_property_readonly("classes_", [](const svc &self) -> py::array {
            PLSSVM_ASSERT(self.data_ != nullptr, "data_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'classes_'" };
            }

            return std::visit([](auto &&data) -> py::array {
                return plssvm::bindings::python::util::vector_to_pyarray(data.classes().value());
            }, *self.data_); }, "The classes labels. ndarray of shape (n_classes,)")
        .def_property_readonly("coef_", [](const svc &self) -> py::array {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'coef_'" };
            }
            if (self.svm_->get_params().kernel_type != plssvm::kernel_function_type::linear) {
                throw py::attribute_error{ "coef_ is only available when using a linear kernel" };
            }

            return std::visit([&](auto &&model) {
                // check if the w ptr has already been set
                if (self.get_w_ptr().empty()) {
                    // score
                    std::ignore = self.svm_->score(model);
                }

                // now, the w ptr is set and can be used
                return py::cast(self.get_w_ptr());
            }, *self.model_); }, "Weights assigned to the features when kernel=\"linear\". ovo: ndarray of shape (n_classes * (n_classes - 1) / 2, n_features). ovr: (n_classes, n_features)")
        .def_property_readonly("dual_coef_", [](const svc &) { throw py::attribute_error{ "'SVC' object has no attribute 'dual_coef_' (not implemented)" }; }, "Dual coefficients of the support vector in the decision function, multiplied by their targets. ndarray of shape (n_classes - 1, n_SV)")
        .def_property_readonly("fit_status_", [](const svc &self) -> int {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'fit_status_'" };
            }

            return 0; }, "0 if correctly fitted, 1 otherwise (will raise exception). int")
        .def_property_readonly("intercept_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'intercept_'" };
            }

            return std::visit([&](auto &&model) {
                std::vector<plssvm::real_type> rho = model.rho();

                // ovr binary special case
                if (self.classification_ == plssvm::classification_type::oaa && model.num_classes() == 2) {
                    rho.pop_back();
                }

                return plssvm::bindings::python::util::vector_to_pyarray(rho);
            }, *self.model_); }, "Constants in decision function. ovo: ndarray of shape (n_classes * (n_classes - 1) / 2,). ovr: ndarray of shape (n_classes,)")
        .def_property_readonly("n_features_in_", [](const svc &self) -> int {
            PLSSVM_ASSERT(self.data_ != nullptr, "data_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'n_features_in_'" };
            }

            return static_cast<int>(std::visit([](auto &&data) { return data.num_features(); }, *self.data_)); }, "Number of features seen during fit. int")
        .def_property_readonly("feature_names_in_", [](const svc &self) -> py::array {
            if (!self.feature_names_.has_value()) {
                throw py::attribute_error{ "'SVC' object has no attribute 'feature_names_in_'" };
            }

            return plssvm::bindings::python::util::vector_to_pyarray(self.feature_names_.value()); }, "Names of features seen during fit. ndarray of shape (n_features_in_,)")
        .def_property_readonly("n_iter_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
            }

            return std::visit([](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(model.num_iters().value()); }, *self.model_); }, "Number of iterations run by the optimization routine to fit the model. ndarray of shape (n_classes * (n_classes - 1) // 2,)")
        .def_property_readonly("support_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
            }

           return plssvm::bindings::python::util::vector_to_pyarray(self.calculate_sv_indices_per_class()); }, "Indices of support vectors. ndarray of shape (n_SV)")
        .def_property_readonly("support_vectors_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_vectors_'" };
            }

            // get the sorted indices
            const std::vector<int> support = self.calculate_sv_indices_per_class();
            // convert support vectors matrix to 2d vector
            std::vector<std::vector<plssvm::real_type>> sv = std::visit([](auto &&model) { return model.support_vectors().to_2D_vector(); }, *self.model_);

            // sort support vectors by their class
            std::vector<std::vector<plssvm::real_type>> sorted_sv{};
            sorted_sv.reserve(sv.size());
            for (const int idx : support) {
                sorted_sv.push_back(std::move(sv[idx]));
            }

            // convert 2D vector back to plssvm::matrix
            return py::cast(plssvm::aos_matrix<plssvm::real_type>{ std::move(sorted_sv) }); }, "Support vectors. ndarray of shape (n_SV, n_features)")
        .def_property_readonly("n_support_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'n_support_'" };
            }

            return std::visit([&](auto &&model) {
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;

                std::map<label_type, std::int32_t> occurrences{};
                // init count map
                for (const label_type &label : model.classes()) {
                    occurrences.insert({ label, std::int32_t{ 0 } });
                }
                // count occurrences
                for (const label_type &label : model.labels()->get()) {
                    ++occurrences[label];
                }
                // convert map values to vector
                std::vector<std::int32_t> n_support{};
                n_support.reserve(occurrences.size());
                for (const auto &[label, n_sv] : occurrences) {
                    n_support.push_back(n_sv);
                }
                // convert to Numpy array
                return plssvm::bindings::python::util::vector_to_pyarray(n_support);
            }, *self.model_); }, "Number of support vectors for each class. ndarray of shape (n_classes,), dtype=int32")
        .def_property_readonly("probA_", [](const svc &) { throw py::attribute_error{ "'SVC' object has no attribute 'probA_' (not implemented)" }; }, "Parameter learned in Platt scaling when probability=True. ndarray of shape (n_classes * (n_classes - 1) / 2)")
        .def_property_readonly("probB_", [](const svc &) { throw py::attribute_error{ "'SVC' object has no attribute 'probB_' (not implemented)" }; }, "Parameter learned in Platt scaling when probability=True. ndarray of shape (n_classes * (n_classes - 1) / 2)")
        .def_property_readonly("shape_fit_", [](const svc &self) {
            PLSSVM_ASSERT(self.data_ != nullptr, "data_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'shape_fit_'" };
            }

            return std::visit([](auto &&data) { return std::make_tuple(static_cast<int>(data.num_data_points()), static_cast<int>(data.num_features())); }, *self.data_); }, "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)")
        .def_property_readonly("_estimator_type", [](const svc &) { return "classifier"; }, "The type of estimator. Always 'classifier' for SVC.");

    //*************************************************************************************************************************************//
    //                                                               METHODS                                                               //
    //*************************************************************************************************************************************//
    py_svc
        .def("decision_function", [](const svc &self, plssvm::soa_matrix<plssvm::real_type> predict_points) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }
            return std::visit([&](auto &&model) -> py::array {
                switch (self.classification_) {
                    case plssvm::classification_type::oaa:
                        {
                            const plssvm::parameter &params = model.get_params();
                            const plssvm::soa_matrix<plssvm::real_type> &sv = model.support_vectors();
                            const plssvm::aos_matrix<plssvm::real_type> &alpha = model.weights().front();  // num_classes x num_data_points
                            const std::vector<plssvm::real_type> &rho = model.rho();
                            plssvm::soa_matrix<plssvm::real_type> w{};  // empty -> no need to befriend the model class!

                            // predict values using OAA -> num_data_points x num_classes
                            // note: must not be const or the custom type_caster won't kick in
                            plssvm::aos_matrix<plssvm::real_type> votes = self.call_predict_values(params, sv, alpha, rho, w, predict_points);

                            // special case for binary classification
                            if (model.num_classes() == 2) {
                                std::vector<plssvm::real_type> reduced_votes(votes.num_rows());
                                for (std::size_t i = 0; i < votes.num_rows(); ++i) {
                                    reduced_votes[i] = -votes(i, 0);
                                }
                                return plssvm::bindings::python::util::vector_to_pyarray(reduced_votes);
                            }
                            return py::cast(votes);
                        }
                    case plssvm::classification_type::oao:
                        {
                            const std::size_t num_features = model.num_features();
                            const std::size_t num_classes = model.num_classes();
                            const std::vector<std::vector<std::size_t>> &index_sets = self.get_index_sets_ptr();

                            const plssvm::parameter &params = model.get_params();
                            const std::vector<plssvm::aos_matrix<plssvm::real_type>> &alpha = model.weights();
                            const std::vector<plssvm::real_type> &rho = model.rho();

                            // create the numpy array
                            py::array_t<plssvm::real_type, py::array::c_style> votes{ { predict_points.num_rows(), plssvm::calculate_number_of_classifiers(plssvm::classification_type::oao, num_classes) } };
                            auto votes_access = votes.mutable_unchecked<2>();

                            // perform one vs. one prediction
                            std::size_t pos = 0;
                            for (std::size_t i = 0; i < num_classes; ++i) {
                                for (std::size_t j = i + 1; j < num_classes; ++j) {
                                    // assemble one vs. one classification matrix and rhs
                                    const std::size_t num_data_points_in_sub_matrix{ index_sets[i].size() + index_sets[j].size() };
                                    const plssvm::aos_matrix<plssvm::real_type> &binary_alpha = alpha[pos];
                                    const std::vector<plssvm::real_type> binary_rho{ rho[pos] };

                                    // create binary support vector matrix, based on the number of classes
                                    const plssvm::soa_matrix<plssvm::real_type> &binary_sv = [&]() {
                                        if (num_classes == 2) {
                                            // no special assembly needed in binary case
                                            return model.support_vectors();
                                        }
                                        // note: if this is changed, it must also be changed in the libsvm_model_parsing.hpp in the calculate_alpha_idx function!!!
                                        // order the indices in increasing order
                                        plssvm::soa_matrix<plssvm::real_type> temp{ plssvm::shape{ num_data_points_in_sub_matrix, num_features }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
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
                                    }();

                                    // we don't use the w optimization for the linear kernel here due to code simplicity
                                    plssvm::soa_matrix<plssvm::real_type> w{};
                                    // predict the values
                                    const plssvm::aos_matrix<plssvm::real_type> binary_votes = self.call_predict_values(params, binary_sv, binary_alpha, binary_rho, w, predict_points);

                                    // update final votes
                                    for (std::size_t pp = 0; pp < predict_points.num_rows(); ++pp) {
                                        votes_access(pp, pos) = binary_votes(pp, 0);
                                    }

                                    // go to next one vs. one classification
                                    ++pos;
                                    // order of the alpha value: 0 vs 1, 0 vs 2, 0 vs 3, 1 vs 2, 1 vs 3, 2 vs 3
                                }
                            }

                            // special case binary classification
                            if (num_classes == 2) {
                                for (std::size_t pp = 0; pp < predict_points.num_rows(); ++pp) {
                                    votes_access(pp, pos) *= plssvm::real_type{ -1.0 };
                                }
                                return votes.reshape(py::array::ShapeContainer{ votes.size() });
                            }
                            return votes;
                        }
                }
                // unreachable
                return py::array{};
            }, *self.model_); }, "Evaluate the decision function for the samples in X.")
        .def("fit", [](svc &self, plssvm::bindings::python::util::soa_matrix_wrapper<plssvm::real_type> data, plssvm::bindings::python::util::label_vector_wrapper<typename svc::possible_vector_types> labels, const std::optional<std::vector<plssvm::real_type>> &sample_weight) -> svc & {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            // sanity check parameter
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }

            // store the used label type
            self.py_dtype_ = labels.dtype;

            // retrieve the potential feature names
            self.feature_names_ = std::move(data.feature_names);

            // create the data set to fit
            std::visit([&](auto &&labels_vector) {
                // get the label type and possible data set types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                using possible_data_set_types = typename svc::possible_data_set_types;
                using possible_model_types = typename svc::possible_model_types;

                // create the data set to fit
                plssvm::classification_data_set<label_type> train_data{ std::move(data.matrix), std::move(labels_vector) };

                // fit the model
                if (self.max_iter_.has_value()) {
                    self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(train_data,
                                                                                        plssvm::epsilon = self.epsilon_,
                                                                                        plssvm::classification = self.classification_,
                                                                                        plssvm::max_iter = self.max_iter_.value()));
                } else {
                    self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(train_data,
                                                                                        plssvm::epsilon = self.epsilon_,
                                                                                        plssvm::classification = self.classification_));
                }

                // store data set internally
                self.data_ = std::make_unique<possible_data_set_types>(std::move(train_data));
            },
                      labels.labels);

            return self; }, py::return_value_policy::reference, "Fit the SVM model according to the given training data.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("get_metadata_routing", [](const svc &) { throw py::attribute_error{ "'SVC' object has no function 'get_metadata_routing' (not implemented)" }; }, "Get metadata routing of this object.")
        .def("get_params", &svc::get_params, "Get parameters for this estimator.", py::arg("deep") = true)
        .def("predict", [](svc &self, plssvm::soa_matrix<plssvm::real_type> data) -> py::array {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            return std::visit([&](auto &&model) {
                // get the label type
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                // create the data set to predict
                const plssvm::classification_data_set<label_type> data_to_predict{ std::move(data) };
                // predict the data
                return plssvm::bindings::python::util::vector_to_pyarray(self.svm_->predict(model, data_to_predict));
            }, *self.model_); }, "Perform classification on samples in X.", py::arg("X"))
        .def("predict_log_proba", [](const svc &, const py::array_t<plssvm::real_type> &) { throw py::attribute_error{ "'SVC' object has no function 'predict_log_proba' (not implemented)" }; }, "Compute log probabilities of possible outcomes for samples in X.", py::arg("X"))
        .def("predict_proba", [](const svc &, const py::array_t<plssvm::real_type> &) { throw py::attribute_error{ "'SVC' object has no function 'predict_proba' (not implemented)" }; }, "Compute probabilities of possible outcomes for samples in X.", py::arg("X"))
        .def("score", [](svc &self, plssvm::soa_matrix<plssvm::real_type> data, plssvm::bindings::python::util::label_vector_wrapper<typename svc::possible_vector_types> labels, const std::optional<std::vector<plssvm::real_type>> &sample_weight) -> plssvm::real_type {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            // sanity check parameter
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            // score the data
            return std::visit([&](auto &&labels_vector) {
                // get the label types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                // create the data set to score
                const plssvm::classification_data_set<label_type> data_to_score{ std::move(data), std::move(labels_vector) };
                // score the data
                try {
                    return self.svm_->score(std::get<plssvm::classification_model<label_type>>(*self.model_), data_to_score);
                } catch (const std::exception &) {
                    throw py::value_error{ fmt::format(R"(The dtype of the labels to score is "{}", but the model was fitted with "{}". Please use the same types for fit and score!)", labels.dtype.attr("name").cast<std::string>(), self.py_dtype_.attr("name").cast<std::string>()) };
                }
            }, labels.labels); }, "Return the mean accuracy on the given test data and labels.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("set_fit_request", [](const svc &) { throw py::attribute_error{ "'SVC' object has no function 'set_fit_request' (not implemented)" }; }, "Request metadata passed to the fit method.")
        .def("set_params", [](svc &self, const py::kwargs &args) -> svc & {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            // check keyword arguments
            plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size", "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state" });

            if (args.contains("C")) {
                self.svm_->set_params(plssvm::cost = args["C"].cast<plssvm::real_type>());
            }
            if (args.contains("kernel")) {
                self.svm_->set_params(plssvm::kernel_type = args["kernel"].cast<plssvm::kernel_function_type>());
            }
            if (args.contains("degree")) {
                self.svm_->set_params(plssvm::degree = args["degree"].cast<int>());
            }
            if (args.contains("gamma")) {
                self.svm_->set_params(plssvm::gamma = args["gamma"].cast<plssvm::gamma_type>());
            }
            if (args.contains("coef0")) {
                self.svm_->set_params(plssvm::coef0 = args["coef0"].cast<plssvm::real_type>());
            }
            if (args.contains("shrinking")) {
                throw py::value_error{ "The 'shrinking' parameter for the 'SVC' is not implemented and makes no sense for a LS-SVM!" };
            }
            if (args.contains("probability")) {
                throw py::value_error{ "The 'probability' parameter for the 'SVC' is not implemented yet!" };
            }
            if (args.contains("tol")) {
                self.epsilon_ = args["tol"].cast<plssvm::real_type>();
            }
            if (args.contains("cache_size")) {
                throw py::value_error{ "The 'cache_size' parameter for the 'SVC' is not implemented and makes no sense for our PLSSVM implementation!" };
            }
            if (args.contains("class_weight")) {
                throw py::value_error{ "The 'class_weight' parameter for the 'SVC' is not implemented yet!" };
            }
            if (args.contains("verbose")) {
                if (args["verbose"].cast<bool>()) {
                    if (plssvm::verbosity == plssvm::verbosity_level::quiet) {
                        // if current verbosity is quiet, override with full verbosity, since 'verbose=TRUE' should never result in no output
                        plssvm::verbosity = plssvm::verbosity_level::full;
                    }
                    // otherwise: use currently active verbosity level
                } else {
                    plssvm::verbosity = plssvm::verbosity_level::quiet;
                }
            }
            if (args.contains("max_iter")) {
                const auto max_iter = args["max_iter"].cast<long long>();
                if (max_iter > 0) {
                    // use provided value
                    self.max_iter_ = static_cast<unsigned long long>(max_iter);
                } else if (max_iter == -1) {
                    // default behavior in PLSSVM
                    self.max_iter_ = std::nullopt;
                } else {
                    // invalid max_iter provided
                    throw py::value_error{ fmt::format("max_iter must either be greater than zero or -1, got {}!", max_iter) };
                }
            }
            if (args.contains("decision_function_shape")) {
                self.classification_ = args["decision_function_shape"].cast<plssvm::classification_type>();
            }
            if (args.contains("break_ties")) {
                throw py::value_error{ "The 'break_ties' parameter for the 'SVC' is not implemented yet!" };
            }
            if (args.contains("random_state")) {
                throw py::value_error{ "The 'random_state' parameter for the 'SVC' is not implemented yet!" };
            }
            return self; }, py::return_value_policy::reference, "Set the parameters of this estimator.")
        .def("set_score_request", [](const svc &) { throw py::attribute_error{ "'SVC' object has no function 'set_score_request' (not implemented)" }; }, "Request metadata passed to the score method.")
        .def("__sklearn_is_fitted__", [](const svc &self) -> bool { return self.model_ != nullptr; }, "Return True if the estimator is fitted, False otherwise.")
        .def("__sklearn_clone__", [](const svc &self) -> svc {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            // create a new SVC instance
            svc new_svc{};
            // copy the parameters
            new_svc.svm_->set_params(self.svm_->get_params());
            new_svc.py_dtype_ = self.py_dtype_;
            new_svc.epsilon_ = self.epsilon_;
            new_svc.max_iter_ = self.max_iter_;
            new_svc.classification_ = self.classification_;
            return new_svc; }, "Clone the estimator.")
        .def("__repr__", [](const svc &self) {
            // get the currently used parameters
            const py::dict used_params = self.get_params(true);
            const py::dict default_params = svc{}.get_params(true);

            std::vector<std::string> non_default_values{};

            // iterate over all available keys and check if the currently used one differs from the default one
            for (auto item : used_params) {
                const auto key = item.first.cast<std::string>();

                // get the values as string
                const std::string used_param_str = py::str(used_params[key.c_str()]);
                const std::string default_param_str = py::str(default_params[key.c_str()]);

                // check if the parameter values are identical, if not, add them to the vector
                if (used_param_str != default_param_str) {
                    if (py::isinstance<py::str>(used_params[key.c_str()])) {
                        non_default_values.push_back(fmt::format("{}='{}'", key, used_param_str));
                    } else {
                        non_default_values.push_back(fmt::format("{}={}", key, used_param_str));
                    }
                }
            }

            return fmt::format("plssvm.svm.SVC({})", fmt::join(non_default_values, ", ")); }, "Print the SVC showing all non-default parameters.")
        .def(py::pickle(
            // clang-format off
            [](const svc &self) {  // __getstate__
                // return a tuple that fully encodes the state of the object
                return py::make_tuple(self.svm_->get_params(), self.epsilon_, self.max_iter_, self.classification_);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 4) {
                    throw std::runtime_error{ "Invalid state!" };
                }
                // create a new C++ instance
                return svc{ t[0].cast<plssvm::parameter>(), t[1].cast<plssvm::real_type>(), t[2].cast<std::optional<unsigned long long>>(), t[3].cast<plssvm::classification_type>() };
            }
            )
             // clang-format on
        );
}

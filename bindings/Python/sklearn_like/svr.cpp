/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/constants.hpp"                     // plssvm::real_type, plssvm::DEFAULT_EPSILON
#include "plssvm/csvm_factory.hpp"                  // plssvm::make_csvr
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/detail/assert.hpp"                 // PLSSVM_ASSERT
#include "plssvm/detail/type_traits.hpp"            // plssvm::detail::remove_cvref_t
#include "plssvm/gamma.hpp"                         // plssvm::gamma_coefficient_type, plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"         // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                        // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/model/regression_model.hpp"        // plssvm::regression_model
#include "plssvm/parameter.hpp"                     // plssvm::parameter, named arguments definition
#include "plssvm/svm/csvr.hpp"                      // plssvm::csvr
#include "plssvm/verbosity_levels.hpp"              // plssvm::verbosity_level, plssvm::verbosity

#include "bindings/Python/bindings_fwd.hpp"                             // forward declare all helper functions to create the Python bindings
#include "bindings/Python/data_set/variant_wrapper.hpp"                 // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/model/variant_wrapper.hpp"                    // plssvm::bindings::python::util::regression_model_wrapper
#include "bindings/Python/type_caster/label_vector_wrapper_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::bindings::python::util::label_vector_wrapper
#include "bindings/Python/type_caster/matrix_type_caster.hpp"           // NOLINT: a custom Pybind11 type caster for a plssvm::matrix
#include "bindings/Python/type_caster/matrix_wrapper_type_caster.hpp"   // a custom Pybind11 type caster for a plssvm::bindings::python::util::matrix_wrapper
#include "bindings/Python/utility.hpp"                                  // plssvm::bindings::python::util::{check_kwargs_for_correctness, vector_to_pyarray}

#include "fmt/format.h"          // fmt::format
#include "fmt/ranges.h"          // fmt::join
#include "pybind11/cast.h"       // py::cast, py::arg
#include "pybind11/numpy.h"      // support for STL types
#include "pybind11/operators.h"  // NOLINT: support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::return_value_policy, py::self, py::dynamic_attr, py::value_error, py::attribute_error, py::tuple, py::pickle
#include "pybind11/pytypes.h"    // py::dict, py::kwargs, py::str
#include "pybind11/stl.h"        // NOLINT: support for STL types

#include <cstdint>    // std::int32_t
#include <exception>  // std::exception
#include <memory>     // std::unique_ptr, std::make_unique
#include <numeric>    // std::iota
#include <optional>   // std::optional, std::nullopt
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <tuple>      // std::make_tuple
#include <utility>    // std::move, std::forward
#include <variant>    // std::holds_alternative
#include <vector>     // std::vector

namespace py = pybind11;

// TODO: implement missing functionality (as far es possible)
/*
 * Currently missing:
 * - shrinking constructor parameter (makes no sense for LS-SVMs)
 * - cache_size constructor parameter (not applicable in PLSSVM)
 * - epsilon constructor parameter (not applicable in PLSSVM since we implement a C-SVR and not an epsilon-SVR)
 * - dual_coef_ attribute
 * - get_metadata_routing function (no idea how to implement this function)
 * - set_fit_request function (no idea how to implement this function)
 * - set_score_request function (no idea how to implement this function)
 * - sample_weight parameter for the fit function
 * - sample_weight parameter for the score function
 */

// dummy
struct svr {
    using possible_vector_types = typename plssvm::bindings::python::util::regression_data_set_wrapper::possible_vector_types;
    using possible_data_set_types = typename plssvm::bindings::python::util::regression_data_set_wrapper::possible_data_set_types;
    using possible_model_types = typename plssvm::bindings::python::util::regression_model_wrapper::possible_model_types;

    /**
     * @brief Construct a default svr wrapper doing nothing.
     */
    svr() :
        svm_{ plssvm::make_csvr(plssvm::gamma = plssvm::gamma_coefficient_type::scale) } { }

    /**
     * @brief Construct a new svr wrapper with the provided parameters.
     * @param[in] params the SVM hyper-parameters
     * @param[in] epsilon the epsilon value for the CG termination criterion
     * @param[in] max_iter the maximum number of CG iterations
     */
    svr(const plssvm::parameter params, const plssvm::real_type epsilon, const std::optional<unsigned long long> max_iter) :
        svm_{ plssvm::make_csvr(params) },
        epsilon_{ epsilon },
        max_iter_{ max_iter } { }

    /**
     * @brief Get the w values used for the coef_ attribute from the currently learned linear model.
     * @return the w values (`[[nodiscard]]`)
     */
    [[nodiscard]] const auto &get_w_ptr() const {
        if (model_ == nullptr) {
            throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
        }
        return std::visit([](auto &&model) -> const auto & { return *model.w_ptr_; }, *model_);
    }

    /**
     * @brief Return the currently used params.
     * @details Necessary for the same Python function and also the string representation.
     * @return a Python dictionary containing the used parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] py::dict get_params(const bool) const {
        const plssvm::parameter params = svm_->get_params();

        // fill a Python dictionary with the supported keys and values
        py::dict py_params;
        py_params["C"] = params.cost;
        // py_params["epsilon"] = 0.1;
        // py_params["cache_size"] = 0;
        py_params["coef0"] = params.coef0;
        py_params["degree"] = params.degree;
        if (std::holds_alternative<plssvm::real_type>(params.gamma)) {
            py_params["gamma"] = std::get<plssvm::real_type>(params.gamma);
        } else {
            // can't use this for both or the numeric value would also be interpreted as a string like '0.001'
            py_params["gamma"] = fmt::format("{}", params.gamma);
        }
        py_params["kernel"] = fmt::format("{}", params.kernel_type);
        py_params["max_iter"] = max_iter_.has_value() ? static_cast<long long>(max_iter_.value()) : -1;
        // py_params["shrinking"] = false;
        py_params["tol"] = epsilon_;
        py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;

        return py_params;
    }

    /// Pointer to the the stored PLSSVM C-SVR instance.
    std::unique_ptr<plssvm::csvr> svm_;
    /// The CG termination criterion if provided.
    plssvm::real_type epsilon_{};
    /// The maximum number of CG iterations if provided.
    std::optional<unsigned long long> max_iter_;

    /// The data type of the labels.
    py::dtype py_dtype_;
    /// Pointer to the regression data set wrapper (represents data sets with all possible label types).
    std::unique_ptr<possible_data_set_types> data_;
    /// Pointer to the regression model wrapper (represents models with all possible label types).
    std::unique_ptr<possible_model_types> model_;

    /// The name of the features. Can only be provided via a Pandas DataFrame.
    std::optional<std::vector<std::string>> feature_names_;
};

void init_sklearn_svr(py::module_ &m) {
    // documentation based on sklearn.svm.SVR documentation
    py::class_<svr> py_svr(m, "SVR", py::dynamic_attr(), "A C-SVR implementation adhering to sklearn.svm.SVR using PLSSVM as backend.");
    py_svr.def(py::init([](const plssvm::kernel_function_type kernel, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0, const plssvm::real_type tol, const plssvm::real_type C, const bool verbose, const long long max_iter) {
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
                   return svr{ params, tol, used_max_iter };
               }),
               "Construct a new SVC classifier.",
               py::kw_only(),
               py::arg("kernel") = plssvm::kernel_function_type::rbf,
               py::arg("degree") = 3,
               py::arg("gamma") = plssvm::gamma_coefficient_type::scale,
               py::arg("coef0") = 0.0,
               py::arg("tol") = 1e-10,
               py::arg("C") = 1.0,
               // py::arg("epsilon") = 0.1,
               // py::arg("shrinking") = true,     // true
               // py::arg("cache_size") = 200,  // 200
               py::arg("verbose") = false,
               py::arg("max_iter") = -1);

    //*************************************************************************************************************************************//
    //                                                             ATTRIBUTES                                                              //
    //*************************************************************************************************************************************//
    py_svr
        .def_property_readonly("coef_", [](const svr &self) -> py::array {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVr' object has no attribute 'coef_'" };
            }
            if (self.svm_->get_params().kernel_type != plssvm::kernel_function_type::linear) {
                throw py::attribute_error{ "coef_ is only available when using a linear kernel" };
            }

            return py::cast(self.get_w_ptr()); }, "Weights assigned to the features when kernel=\"linear\". ndarray of shape (n_features, n_classes)")
        .def_property_readonly("dual_coef_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'dual_coef_' (not implemented)" }; }, "Dual coefficients of the support vector in the decision function. ndarray of shape (1, n_SV)")
        .def_property_readonly("fit_status_", [](const svr &self) -> int {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'fit_status_'" };
            }

            return 0; }, "0 if correctly fitted, 1 otherwise (will raise exception). int")
        .def_property_readonly("intercept_", [](const svr &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'intercept_'" };
            }

            return std::visit([&](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(model.rho()); }, *self.model_); }, "Constants in decision function. ndarray of shape (1,)")
        .def_property_readonly("n_features_in_", [](const svr &self) -> int {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'n_features_in_'" };
            }

            return static_cast<int>(std::visit([](auto &&data) { return data.num_features(); }, *self.data_)); }, "Number of features seen during fit. int")
        .def_property_readonly("feature_names_in_", [](const svr &self) -> py::array {
            if (!self.feature_names_.has_value()) {
                throw py::attribute_error{ "'SVR' object has no attribute 'feature_names_in_'" };
            }

            return plssvm::bindings::python::util::vector_to_pyarray(self.feature_names_.value()); }, "Names of features seen during fit. ndarray of shape (n_features_in_,)")
        .def_property_readonly("n_iter_", [](const svr &self) -> int {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'support_'" };
            }

            return std::visit([](auto &&model) { return static_cast<int>(model.num_iters().value().front()); }, *self.model_); }, "Number of iterations run by the optimization routine to fit the model. int")
        .def_property_readonly("support_", [](const svr &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'support_'" };
            }

            // for the SVR, the indices do not need to be sorted
            std::vector<int> support(std::visit([](auto &&model) { return model.num_support_vectors(); }, *self.model_));
            std::iota(support.begin(), support.end(), 0);
            return plssvm::bindings::python::util::vector_to_pyarray(support); }, "Indices of support vectors. ndarray of shape (n_SV)")
        .def_property_readonly("support_vectors_", [](const svr &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'support_vectors_'" };
            }

            // for the SVR, the support vectors do not need to be sorted
            return std::visit([](auto &&model) { return py::cast(model.support_vectors()); }, *self.model_); }, "Support vectors. ndarray of shape (n_SV, n_features)")
        .def_property_readonly("n_support_", [](const svr &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'n_support_'" };
            }

            return std::visit([](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(std::vector<std::int32_t>{ static_cast<std::int32_t>(model.num_support_vectors()) }); }, *self.model_); }, "Number of support vectors for each class. ndarray of shape (1,), dtype=int32")
        .def_property_readonly("shape_fit_", [](const svr &self) {
            PLSSVM_ASSERT(self.data_ != nullptr, "data_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'shape_fit_'" };
            }

            return std::visit([](auto &&data) { return std::make_tuple(static_cast<int>(data.num_data_points()), static_cast<int>(data.num_features())); }, *self.data_); }, "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)")
        .def_property_readonly("_estimator_type", [](const svr &) { return "regressor"; }, "The type of estimator. Always 'regressor' for SVR.");

    //*************************************************************************************************************************************//
    //                                                               METHODS                                                               //
    //*************************************************************************************************************************************//
    py_svr
        .def("fit", [](svr &self, plssvm::bindings::python::util::soa_matrix_wrapper<plssvm::real_type> data, plssvm::bindings::python::util::label_vector_wrapper<typename svr::possible_vector_types> labels, const std::optional<std::vector<plssvm::real_type>> &sample_weight) -> svr & {
           PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
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
                using possible_data_set_types = typename svr::possible_data_set_types;
                using possible_model_types = typename svr::possible_model_types;

                // create the data set to fit
                plssvm::regression_data_set<label_type> train_data{ std::move(data.matrix), std::move(labels_vector) };

                // fit the model using potentially provided keyword arguments
                if (self.max_iter_.has_value()) {
                    self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(train_data,
                                                                                        plssvm::epsilon = self.epsilon_,
                                                                                        plssvm::max_iter = self.max_iter_.value()));
                } else {
                    self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(train_data, plssvm::epsilon = self.epsilon_));
                }

                // store data set internally
                self.data_ = std::make_unique<possible_data_set_types>(std::move(train_data));
            },
                      labels.labels);

            return self; }, py::return_value_policy::reference, "Fit the SVM model according to the given training data.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("get_metadata_routing", [](const svr &) { throw py::attribute_error{ "'SVR' object has no function 'get_metadata_routing' (not implemented)" }; }, "Get metadata routing of this object.")
        .def("get_params", &svr::get_params, "Get parameters for this estimator.", py::arg("deep") = true)
        .def("predict", [](svr &self, plssvm::soa_matrix<plssvm::real_type> data) -> py::array {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            return std::visit([&](auto &&model) {
                // get the label type
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                // create the data set to predict
                const plssvm::regression_data_set<label_type> data_to_predict{ std::move(data) };
                // predict the data
                return plssvm::bindings::python::util::vector_to_pyarray(self.svm_->predict(model, data_to_predict));
            }, *self.model_); }, "Perform classification on samples in X.", py::arg("X"))
        .def("score", [](svr &self, plssvm::soa_matrix<plssvm::real_type> data, plssvm::bindings::python::util::label_vector_wrapper<typename svr::possible_vector_types> labels, const std::optional<std::vector<plssvm::real_type>> &sample_weight) -> plssvm::real_type {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            // score the data
            return std::visit([&](auto &&labels_vector) {
                // get the label types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                // create the data set to score
                const plssvm::regression_data_set<label_type> data_to_score{ std::move(data), std::move(labels_vector) };
                // score the data
                try {
                    return self.svm_->score(std::get<plssvm::regression_model<label_type>>(*self.model_), data_to_score);
                } catch (const std::exception &) {
                    throw py::value_error{ fmt::format("The dtype of the labels to score is \"{}\", but the model was fitted with \"{}\". Please use the same types for fit and score!", labels.dtype.attr("name").cast<std::string>(), self.py_dtype_.attr("name").cast<std::string>()) };
                }
            }, labels.labels); }, "Return the mean accuracy on the given test data and labels.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("set_fit_request", [](const svr &) { throw py::attribute_error{ "'SVR' object has no function 'set_fit_request' (not implemented)" }; }, "Request metadata passed to the fit method.")
        .def("set_params", [](svr &self, const py::kwargs &args) -> svr & {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            // check keyword arguments
            plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "tol", "cache_size", "verbose", "max_iter", "epsilon" });

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
            if (args.contains("tol")) {
                self.epsilon_ = args["tol"].cast<plssvm::real_type>();
            }
            if (args.contains("C")) {
                self.svm_->set_params(plssvm::cost = args["C"].cast<plssvm::real_type>());
            }
            if (args.contains("epsilon")) {
                throw py::value_error{ "The 'epsilon' parameter for the 'SVR' is not implemented yet!" };
            }
            if (args.contains("shrinking")) {
                throw py::value_error{ "The 'shrinking' parameter for the 'SVR' is not implemented yet!" };
            }
            if (args.contains("cache_size")) {
                throw py::value_error{ "The 'cache_size' parameter for the 'SVR' is not implemented yet!" };
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
                    // default behavior in PLSSVM -> do nothing
                } else {
                    // invalid max_iter provided
                    throw py::value_error{ fmt::format("max_iter must either be greater than zero or -1, got {}!", max_iter) };
                }
            }
            return self; }, py::return_value_policy::reference, "Set the parameters of this estimator.")
        .def("set_score_request", [](const svr &) { throw py::attribute_error{ "'SVR' object has no function 'set_score_request' (not implemented)" }; }, "Request metadata passed to the score method.")
        .def("__sklearn_is_fitted__", [](const svr &self) -> bool { return self.model_ != nullptr; }, "Return True if the estimator is fitted, False otherwise.")
        .def("__sklearn_clone__", [](const svr &self) -> svr {
            PLSSVM_ASSERT(self.svm_ != nullptr, "svm_ may not be a nullptr! Maybe you forgot to initialize it?");
            // create a new SVR instance
            svr new_svr{};
            // copy the parameters
            new_svr.svm_->set_params(self.svm_->get_params());
            new_svr.py_dtype_ = self.py_dtype_;
            new_svr.epsilon_ = self.epsilon_;
            new_svr.max_iter_ = self.max_iter_;
            return new_svr; }, "Clone the estimator.")
        .def("__repr__", [](const svr &self) {
            // get the currently used parameters
            const py::dict used_params = self.get_params(true);
            const py::dict default_params = svr{}.get_params(true);

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

            return fmt::format("plssvm.svm.SVR({})", fmt::join(non_default_values, ", ")); }, "Print the SVR showing all non-default parameters.")
        .def(py::pickle(
            // clang-format off
            [](const svr &self) {  // __getstate__
                // return a tuple that fully encodes the state of the object
                return py::make_tuple(self.svm_->get_params(), self.epsilon_, self.max_iter_);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 3) {
                    throw std::runtime_error{ "Invalid state!" };
                }
                // create a new C++ instance
                return svr{ t[0].cast<plssvm::parameter>(), t[1].cast<plssvm::real_type>(), t[2].cast<std::optional<unsigned long long>>() };
            }
            )
             // clang-format on
        );
    ;
}

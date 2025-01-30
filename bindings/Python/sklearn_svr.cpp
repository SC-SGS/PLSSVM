/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/constants.hpp"                     // plssvm::real_type
#include "plssvm/csvm_factory.hpp"                  // plssvm::make_csvr
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/detail/type_traits.hpp"            // plssvm::detail::remove_cvref_t
#include "plssvm/gamma.hpp"                         // plssvm::gamma_coefficient_type, plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"         // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                        // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/model/regression_model.hpp"        // plssvm::regression_model
#include "plssvm/parameter.hpp"                     // plssvm::parameter, named arguments definition
#include "plssvm/svm/csvr.hpp"                      // plssvm::csvr
#include "plssvm/verbosity_levels.hpp"              // plssvm::verbosity_level, plssvm::verbosity

#include "bindings/Python/conversion_from_python.hpp"    // plssvm::bindings::python::util::{pyobject_to_vector, pyobject_to_matrix}
#include "bindings/Python/conversion_to_python.hpp"      // plssvm::bindings::python::util::{vector_to_pyarray, matrix_to_pyarray}
#include "bindings/Python/data_set/variant_wrapper.hpp"  // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/model/variant_wrapper.hpp"     // plssvm::bindings::python::util::regression_model_wrapper
#include "bindings/Python/utility.hpp"                   // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_gamma_kwarg_to_variant}

#include "fmt/format.h"          // fmt::format
#include "pybind11/numpy.h"      // support for STL types
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self, py::dynamic_attr, py::value_error, py::attribute_error
#include "pybind11/stl.h"        // support for STL types

#include <cstdint>   // std::int32_t
#include <memory>    // std::unique_ptr, std::make_unique
#include <numeric>   // std::iota
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <tuple>     // std::make_tuple
#include <utility>   // std::move
#include <variant>   // std::holds_alternative
#include <vector>    // std::vector

namespace py = pybind11;

// TODO: implement missing functionality (as far es possible)

// dummy
struct svr {
    using possible_vector_types = typename plssvm::bindings::python::util::regression_data_set_wrapper::possible_vector_types;
    using possible_data_set_types = typename plssvm::bindings::python::util::regression_data_set_wrapper::possible_data_set_types;
    using possible_model_types = typename plssvm::bindings::python::util::regression_model_wrapper::possible_model_types;

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
        py_params["cache_size"] = 0;
        py_params["coef0"] = params.coef0;
        py_params["degree"] = params.degree;
        if (std::holds_alternative<plssvm::real_type>(params.gamma)) {
            py_params["gamma"] = std::get<plssvm::real_type>(params.gamma);
        } else {
            switch (std::get<plssvm::gamma_coefficient_type>(params.gamma)) {
                case plssvm::gamma_coefficient_type::automatic:
                    py_params["gamma"] = "auto";
                    break;
                case plssvm::gamma_coefficient_type::scale:
                    py_params["gamma"] = "scale";
                    break;
            }
        }
        py_params["kernel"] = fmt::format("{}", params.kernel_type);
        py_params["max_iter"] = max_iter_.has_value() ? static_cast<long long>(max_iter_.value()) : -1;
        py_params["shrinking"] = false;
        py_params["tol"] = epsilon_.value_or(plssvm::real_type{ 1e-10 });
        py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;

        return py_params;
    }

    py::dtype py_dtype_{};
    std::optional<plssvm::real_type> epsilon_{};
    std::optional<unsigned long long> max_iter_{};

    std::unique_ptr<plssvm::csvr> svm_{ plssvm::make_csvr() };
    std::unique_ptr<possible_data_set_types> data_{};
    std::unique_ptr<possible_model_types> model_{};

    std::optional<std::vector<std::string>> feature_names_{};
};

namespace {

void parse_provided_kwargs(svr &self, const py::kwargs &args) {
    // check keyword arguments
    plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "tol", "cache_size", "verbose", "max_iter", "epsilon" });

    if (args.contains("C")) {
        self.svm_->set_params(plssvm::cost = args["C"].cast<plssvm::real_type>());
    }
    if (args.contains("kernel")) {
        const auto kernel_str = args["kernel"].cast<std::string>();
        plssvm::kernel_function_type kernel{};
        if (kernel_str == "linear") {
            kernel = plssvm::kernel_function_type::linear;
        } else if (kernel_str == "poly") {
            kernel = plssvm::kernel_function_type::polynomial;
        } else if (kernel_str == "rbf") {
            kernel = plssvm::kernel_function_type::rbf;
        } else if (kernel_str == "sigmoid") {
            kernel = plssvm::kernel_function_type::sigmoid;
        } else if (kernel_str == "laplacian") {
            kernel = plssvm::kernel_function_type::laplacian;
        } else if (kernel_str == "chi_squared") {
            kernel = plssvm::kernel_function_type::chi_squared;
        } else if (kernel_str == "precomputed") {
            throw py::value_error{ R"(The "kernel = 'precomputed'" parameter for the 'SVR' is not implemented yet!)" };
        } else {
            throw py::value_error{ fmt::format("'{}' is not in list", kernel_str) };
        }
        self.svm_->set_params(plssvm::kernel_type = kernel);
    }
    if (args.contains("degree")) {
        self.svm_->set_params(plssvm::degree = args["degree"].cast<int>());
    }
    if (args.contains("gamma")) {
        const plssvm::gamma_type gamma = plssvm::bindings::python::util::convert_gamma_kwarg_to_variant(args);
        if (std::holds_alternative<plssvm::real_type>(gamma)) {
            self.svm_->set_params(plssvm::gamma = std::get<plssvm::real_type>(gamma));
        } else {
            self.svm_->set_params(plssvm::gamma = std::get<plssvm::gamma_coefficient_type>(gamma));
        }
    }
    if (args.contains("coef0")) {
        self.svm_->set_params(plssvm::coef0 = args["coef0"].cast<plssvm::real_type>());
    }
    if (args.contains("shrinking")) {
        throw py::value_error{ "The 'shrinking' parameter for the 'SVR' is not implemented yet!" };
    }
    if (args.contains("tol")) {
        self.epsilon_ = args["tol"].cast<plssvm::real_type>();
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
    if (args.contains("epsilon")) {
        throw py::value_error{ "The 'epsilon' parameter for the 'SVR' is not implemented yet!" };
    }
}

void fit(svr &self) {
    // perform sanity checks
    if (self.svm_->get_params().cost <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "C <= 0" };
    }
    if (self.svm_->get_params().degree < 0) {
        throw py::value_error{ "degree of polynomial kernel < 0" };
    }
    if (self.epsilon_.has_value() && self.epsilon_.value() <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "eps <= 0" };
    }

    // fit the model using potentially provided keyword arguments
    std::visit([&](auto &&data) {
        using possible_model_types = typename svr::possible_model_types;

        if (self.epsilon_.has_value() && self.max_iter_.has_value()) {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::epsilon = self.epsilon_.value(),
                                                                                plssvm::max_iter = self.max_iter_.value()));
        } else if (self.epsilon_.has_value()) {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::epsilon = self.epsilon_.value()));
        } else if (self.max_iter_.has_value()) {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::max_iter = self.max_iter_.value()));
        } else {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data));
        }
    },
               *self.data_);
}

}  // namespace

void init_sklearn_svr(py::module_ &m) {
    // documentation based on sklearn.svm.SVR documentation
    py::class_<svr> py_svr(m, "SVR", py::dynamic_attr());
    py_svr.def(py::init([](const py::kwargs &args) {
                   // to silence constructor messages
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
                   } else {
                       // sklearn default is quiet
                       plssvm::verbosity = plssvm::verbosity_level::quiet;
                   }

                   // create SVR class
                   auto self = std::make_unique<svr>();
                   parse_provided_kwargs(*self, args);
                   return self;
               }),
               "Construct a new SVR classifier.");

    //*************************************************************************************************************************************//
    //                                                             ATTRIBUTES                                                              //
    //*************************************************************************************************************************************//
    py_svr
        .def_property_readonly("coef_", [](const svr &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVr' object has no attribute 'coef_'" };
            }
            if (self.svm_->get_params().kernel_type != plssvm::kernel_function_type::linear) {
                throw py::attribute_error{ "coef_ is only available when using a linear kernel" };
            }

            return plssvm::bindings::python::util::matrix_to_pyarray(self.get_w_ptr()); }, "Weights assigned to the features when kernel=\"linear\". ndarray of shape (n_features, n_classes)")
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
        .def_property_readonly("feature_names_in_", [](const svr &self) {
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
            return std::visit([](auto &&model) { return plssvm::bindings::python::util::matrix_to_pyarray(model.support_vectors()); }, *self.model_); }, "Support vectors. ndarray of shape (n_SV, n_features)")
        .def_property_readonly("n_support_", [](const svr &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'n_support_'" };
            }

            return std::visit([](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(std::vector<std::int32_t>{ static_cast<std::int32_t>(model.num_support_vectors()) }); }, *self.model_); }, "Number of support vectors for each class. ndarray of shape (1,), dtype=int32")
        .def_property_readonly("shape_fit_", [](const svr &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'shape_fit_'" };
            }

            return std::visit([](auto &&data) { return std::make_tuple(static_cast<int>(data.num_data_points()), static_cast<int>(data.num_features())); }, *self.data_); }, "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)")
        .def_property_readonly("_estimator_type", [](const svr &) { return "regressor"; }, "The type of estimator. Always 'regressor' for SVR.");

    //*************************************************************************************************************************************//
    //                                                               METHODS                                                               //
    //*************************************************************************************************************************************//
    py_svr
        .def("fit", [](svr &self, const py::object &data, const py::object &labels, const std::optional<std::vector<plssvm::real_type>> &sample_weight) -> svr & {
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }

            // convert the labels to a std::vector
            const auto &[labels_vector_variant, dtype] = plssvm::bindings::python::util::pyobject_to_vector<typename svr::possible_vector_types>(labels);
            self.py_dtype_ = dtype;

            // convert the data py::object to a plssvm::aos_matrix
            const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);
            self.feature_names_ = opt_feature_names;

            // create the data set to fit
            std::visit([&self, &data_matrix = data_matrix](auto &&labels_vector) {
                // get the label type and possible data set types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                using possible_data_set_types = typename svr::possible_data_set_types;
                // create the data set to fit
                self.data_ = std::make_unique<possible_data_set_types>(plssvm::regression_data_set<label_type>(data_matrix, labels_vector));
            },
                       labels_vector_variant);

            // fit the model using potentially provided keyword arguments
            fit(self);
            return self; }, "Fit the SVM model according to the given training data.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt, py::return_value_policy::reference)
        .def("get_metadata_routing", [](const svr &) { throw py::attribute_error{ "'SVR' object has no function 'get_metadata_routing' (not implemented)" }; }, "Get metadata routing of this object.")
        .def("get_params", &svr::get_params, "Get parameters for this estimator.", py::arg("deep") = true)
        .def("predict", [](svr &self, const py::object &data) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            // convert the data py::object to a plssvm::aos_matrix
            const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);

            return std::visit([&self, &data_matrix = data_matrix](auto &&model) {
                // get the label type
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                // create the data set to predict
                const plssvm::regression_data_set<label_type> data_to_predict{ data_matrix };
                // predict the data
                return plssvm::bindings::python::util::vector_to_pyarray(self.svm_->predict(model, data_to_predict));
            }, *self.model_); }, "Perform classification on samples in X.")
        .def("score", [](svr &self, const py::object &data, const py::object &labels, const std::optional<std::vector<plssvm::real_type>> &sample_weight) -> plssvm::real_type {
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            // convert the labels to a std::vector
            const auto &[labels_vector_variant, dtype] = plssvm::bindings::python::util::pyobject_to_vector<typename svr::possible_vector_types>(labels);

            // convert the data py::object to a plssvm::aos_matrix
            const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);

            // score the data
            return std::visit([&self, &data_matrix = data_matrix, &dtype = dtype](auto &&labels_vector) {
                // get the label types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                // create the data set to score
                const plssvm::regression_data_set<label_type> data_to_score{ data_matrix, labels_vector };
                // score the data
                try {
                    return self.svm_->score(std::get<plssvm::regression_model<label_type>>(*self.model_), data_to_score);
                } catch (const std::exception &) {
                    throw py::value_error{ fmt::format("The dtype of the labels to score is \"{}\", but the model was fitted with \"{}\". Please use the same types for fit and score!", dtype.attr("name").cast<std::string>(), self.py_dtype_.attr("name").cast<std::string>()) };
                }
            }, labels_vector_variant); }, "Return the mean accuracy on the given test data and labels.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("set_fit_request", [](const svr &) { throw py::attribute_error{ "'SVR' object has no function 'set_fit_request' (not implemented)" }; }, "Request metadata passed to the fit method.")
        .def("set_params", [](svr &self, const py::kwargs &args) -> svr & {
            parse_provided_kwargs(self, args);
            return self; }, "Set the parameters of this estimator.", py::return_value_policy::reference)
        .def("set_score_request", [](const svr &) { throw py::attribute_error{ "'SVR' object has no function 'set_score_request' (not implemented)" }; }, "Request metadata passed to the score method.")
        .def("__sklearn_is_fitted__", [](const svr &self) -> bool { return self.model_ != nullptr; }, "Return True if the estimator is fitted, False otherwise.")
        .def("__sklearn_clone__", [](const svr &self) -> svr {
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
            py::dict used_params = self.get_params(true);
            py::dict default_params = svr{}.get_params(true);

            std::vector<std::string> non_default_values{};

            // iterate over all available keys and check if the currently used one differs from the default one
            for (auto item : used_params) {
                const auto key = item.first.cast<std::string>();

                // get the values as string
                const std::string used_param_str = py::str(used_params[key.c_str()]);
                const std::string default_param_str = py::str(default_params[key.c_str()]);

                // check if the parameter values are identical, if not, add them to the vector
                if (used_param_str != default_param_str) {
                    non_default_values.push_back(fmt::format("{}={}", key, used_param_str));
                }
            }

            return fmt::format("plssvm.SVR({})", fmt::join(non_default_values, ", ")); }, "Print the SVR showing all non-default parameters.");
}

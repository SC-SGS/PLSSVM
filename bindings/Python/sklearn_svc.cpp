/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/classification_types.hpp"              // plssvm::classification_type
#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/csvm_factory.hpp"                      // plssvm::make_csvc
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/detail/type_traits.hpp"                // plssvm::detail::remove_cvref_t
#include "plssvm/gamma.hpp"                             // plssvm::gamma_coefficient_type, plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"             // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                            // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/model/classification_model.hpp"        // plssvm::classification_model
#include "plssvm/parameter.hpp"                         // plssvm::parameter, named arguments definition
#include "plssvm/svm/csvc.hpp"                          // plssvm::csvc
#include "plssvm/verbosity_levels.hpp"                  // plssvm::verbosity_level, plssvm::verbosity

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, pyarray_t_to_vector, pyarray_to_matrix}

#include "fmt/format.h"          // fmt::format
#include "pybind11/numpy.h"      // support for STL types
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self, py::dynamic_attr
#include "pybind11/stl.h"        // support for STL types

#include <algorithm>  // std::fill
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int32_t
#include <cstdint>    // fixed-width integers
#include <exception>  // std::exception
#include <map>        // std::map
#include <memory>     // std::unique_ptr, std::make_unique
#include <optional>   // std::optional, std::nullopt
#include <string>     // std::string
#include <tuple>      // std::make_tuple
#include <utility>    // std::move
#include <variant>    // std::holds_alternative, std::variant, std::visit
#include <vector>     // std::vector

namespace py = pybind11;

// TODO: implement missing functionality (as far es possible)

// dummy
struct svc {
    using possible_data_set_types = std::variant<plssvm::classification_data_set<bool>,           // np.bool
                                                 plssvm::classification_data_set<std::int8_t>,    // np.int8
                                                 plssvm::classification_data_set<std::uint8_t>,   // np.uint8
                                                 plssvm::classification_data_set<std::int16_t>,   // np.int16
                                                 plssvm::classification_data_set<std::uint16_t>,  // np.uint16
                                                 plssvm::classification_data_set<std::int32_t>,   // np.int32
                                                 plssvm::classification_data_set<std::uint32_t>,  // np.uint32
                                                 plssvm::classification_data_set<std::int64_t>,   // np.int64
                                                 plssvm::classification_data_set<std::uint64_t>,  // np.uint64
                                                 plssvm::classification_data_set<float>,          // np.float32
                                                 plssvm::classification_data_set<double>,         // np.float64
                                                 plssvm::classification_data_set<std::string>>;   // np.str

    using possible_model_types = std::variant<plssvm::classification_model<bool>,           // np.bool
                                              plssvm::classification_model<std::int8_t>,    // np.int8
                                              plssvm::classification_model<std::uint8_t>,   // np.uint8
                                              plssvm::classification_model<std::int16_t>,   // np.int16
                                              plssvm::classification_model<std::uint16_t>,  // np.uint16
                                              plssvm::classification_model<std::int32_t>,   // np.int32
                                              plssvm::classification_model<std::uint32_t>,  // np.uint32
                                              plssvm::classification_model<std::int64_t>,   // np.int64
                                              plssvm::classification_model<std::uint64_t>,  // np.uint64
                                              plssvm::classification_model<float>,          // np.float32
                                              plssvm::classification_model<double>,         // np.float64
                                              plssvm::classification_model<std::string>>;   // np.str

    // wrapper class to make friendship work
    template <typename... Args>
    auto call_predict_values(Args &&...args) const {
        return svm_->predict_values(std::forward<Args>(args)...);
    }

    py::dtype py_dtype{};
    std::optional<plssvm::real_type> epsilon{};
    std::optional<unsigned long long> max_iter{};
    plssvm::classification_type classification{ plssvm::classification_type::oaa };

    std::unique_ptr<plssvm::csvc> svm_{ plssvm::make_csvc(plssvm::gamma = plssvm::gamma_coefficient_type::scale) };
    std::unique_ptr<possible_data_set_types> data_{};
    std::unique_ptr<possible_model_types> model_{};

    std::optional<std::vector<std::string>> feature_names_{};
};

namespace {

void parse_provided_kwargs(svc &self, const py::kwargs &args) {
    // check keyword arguments
    plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size", "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state" });

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
            throw py::attribute_error{ R"(The "kernel = 'precomputed'" parameter for the 'SVC' is not implemented yet!)" };
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
        throw py::attribute_error{ "The 'shrinking' parameter for the 'SVC' is not implemented yet!" };
    }
    if (args.contains("probability")) {
        throw py::attribute_error{ "The 'probability' parameter for the 'SVC' is not implemented yet!" };
    }
    if (args.contains("tol")) {
        self.epsilon = args["tol"].cast<plssvm::real_type>();
    }
    if (args.contains("cache_size")) {
        throw py::attribute_error{ "The 'cache_size' parameter for the 'SVC' is not implemented yet!" };
    }
    if (args.contains("class_weight")) {
        throw py::attribute_error{ "The 'class_weight' parameter for the 'SVC' is not implemented yet!" };
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
            self.max_iter = static_cast<unsigned long long>(max_iter);
        } else if (max_iter == -1) {
            // default behavior in PLSSVM -> do nothing
        } else {
            // invalid max_iter provided
            throw py::value_error{ fmt::format("max_iter must either be greater than zero or -1, got {}!", max_iter) };
        }
    }
    if (args.contains("decision_function_shape")) {
        const std::string &dfs = args["decision_function_shape"].cast<std::string>();
        if (dfs == "ovo") {
            self.classification = plssvm::classification_type::oao;
        } else if (dfs == "ovr") {
            self.classification = plssvm::classification_type::oaa;
        } else {
            throw py::value_error{ fmt::format("decision_function_shape must be either 'ovr' or 'ovo', got {}.", dfs) };
        }
    }
    if (args.contains("break_ties")) {
        throw py::attribute_error{ "The 'break_ties' parameter for the 'SVC' is not implemented yet!" };
    }
    if (args.contains("random_state")) {
        throw py::attribute_error{ "The 'random_state' parameter for the 'SVC' is not implemented yet!" };
    }
}

void fit(svc &self) {
    // perform sanity checks
    if (self.svm_->get_params().cost <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "C <= 0" };
    }
    if (self.svm_->get_params().degree < 0) {
        throw py::value_error{ "degree of polynomial kernel < 0" };
    }
    if (self.epsilon.has_value() && self.epsilon.value() <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "eps <= 0" };
    }

    // fit the model using potentially provided keyword arguments
    std::visit([&](auto &&data) {
        using possible_model_types = typename svc::possible_model_types;

        if (self.epsilon.has_value() && self.max_iter.has_value()) {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::classification = self.classification,
                                                                                plssvm::epsilon = self.epsilon.value(),
                                                                                plssvm::max_iter = self.max_iter.value()));
        } else if (self.epsilon.has_value()) {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::classification = self.classification,
                                                                                plssvm::epsilon = self.epsilon.value()));
        } else if (self.max_iter.has_value()) {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::classification = self.classification,
                                                                                plssvm::max_iter = self.max_iter.value()));
        } else {
            self.model_ = std::make_unique<possible_model_types>(self.svm_->fit(data,
                                                                                plssvm::classification = self.classification));
        }
    },
               *self.data_);
}

template <typename svc>
[[nodiscard]] std::vector<int> calculate_sv_indices_per_class(const svc &self) {
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
                      *self.model_);
}

}  // namespace

void init_sklearn_svc(py::module_ &m) {
    // documentation based on sklearn.svm.SVC documentation
    py::class_<svc> py_svc(m, "SVC", py::dynamic_attr());
    py_svc.def(py::init([](const py::kwargs &args) {
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

                   // create SVC class
                   auto self = std::make_unique<svc>();
                   parse_provided_kwargs(*self, args);
                   return self;
               }),
               "Construct a new SVC classifier.");

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
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'classes_'" };
            }

            return std::visit([](auto &&data) -> py::array {
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type;
                return plssvm::bindings::python::util::vector_to_pyarray<label_type>(data.classes().value());
            }, *self.data_); }, "The classes labels. ndarray of shape (n_classes,)")
        .def_property_readonly("coef_", [](const svc &) { throw py::attribute_error{ "'SVC' object has no attribute 'coef_' (not implemented)" }; }, "Weights assigned to the features when kernel=\"linear\". ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)")
        .def_property_readonly("dual_coef_", [](const svc &) { throw py::attribute_error{ "'SVC' object has no attribute 'dual_coef_' (not implemented)" }; }, "Dual coefficients of the support vector in the decision function, multiplied by their targets. ndarray of shape (n_classes - 1, n_SV)")
        .def_property_readonly("fit_status_", [](const svc &self) -> int {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'fit_status_'" };
            }

            return 0; }, "0 if correctly fitted, 1 otherwise (will raise exception). int")
        .def_property_readonly("intercept_", [](const svc &) { throw py::attribute_error{ "'SVC' object has no attribute 'intercept_' (not implemented)" }; }, "Constants in decision function. ndarray of shape (n_classes * (n_classes - 1) / 2,)")
        .def_property_readonly("n_features_in_", [](const svc &self) -> int {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'n_features_in_'" };
            }

            return static_cast<int>(std::visit([](auto &&data) { return data.num_features(); }, *self.data_)); }, "Number of features seen during fit. int")
        .def_property_readonly("feature_names_in_", [](const svc &self) {
            if (!self.feature_names_.has_value()) {
                throw py::attribute_error{ "'SVC' object has no attribute 'feature_names_in_'" };
            }

            return plssvm::bindings::python::util::vector_to_pyarray(self.feature_names_.value()); }, "Names of features seen during fit. ndarray of shape (n_features_in_,)")
        .def_property_readonly("n_iter_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
            }

            return plssvm::bindings::python::util::vector_to_pyarray(std::visit([](auto &&model) { return model.num_iters().value(); }, *self.model_)); }, "Number of iterations run by the optimization routine to fit the model. ndarray of shape (n_classes * (n_classes - 1) // 2,)")
        .def_property_readonly("support_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
            }

            return plssvm::bindings::python::util::vector_to_pyarray(calculate_sv_indices_per_class(self)); }, "Indices of support vectors. ndarray of shape (n_SV)")
        .def_property_readonly("support_vectors_", [](const svc &self) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_vectors_'" };
            }

            // get the sorted indices
            const std::vector<int> support = calculate_sv_indices_per_class(self);
            // convert support vectors matrix to 2d vector
            std::vector<std::vector<plssvm::real_type>> sv = std::visit([](auto &&model) { return model.support_vectors().to_2D_vector(); }, *self.model_);

            // sort support vectors by their class
            std::vector<std::vector<plssvm::real_type>> sorted_sv{};
            sorted_sv.reserve(sv.size());
            for (const int idx : support) {
                sorted_sv.push_back(std::move(sv[idx]));
            }

            // convert 2D vector back to plssvm::matrix
            return plssvm::bindings::python::util::matrix_to_pyarray(plssvm::aos_matrix<plssvm::real_type>{ std::move(sorted_sv) }); }, "Support vectors. ndarray of shape (n_SV, n_features)")
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
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'shape_fit_'" };
            }

            return std::visit([](auto &&data) { return std::make_tuple(static_cast<int>(data.num_data_points()), static_cast<int>(data.num_features())); }, *self.data_); }, "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)")
        .def_property_readonly("_estimator_type", [](const svc &) { return "classifier"; }, "The type of estimator. Always 'classifier' for SVC.");

    //*************************************************************************************************************************************//
    //                                                               METHODS                                                               //
    //*************************************************************************************************************************************//
    py_svc
        .def("decision_function", [](const svc &self, py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> predict_points) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }
            if (self.classification != plssvm::classification_type::oaa) {
                throw py::attribute_error{ "The \"decision_function\" is currently only supported for ovr!" };
            }

            return std::visit([&](auto &&model) -> py::array {
                const plssvm::parameter &params = model.get_params();
                const plssvm::soa_matrix<plssvm::real_type> &sv = model.support_vectors();
                const plssvm::aos_matrix<plssvm::real_type> &alpha = model.weights().front();  // num_classes x num_data_points
                const std::vector<plssvm::real_type> &rho = model.rho();
                plssvm::soa_matrix<plssvm::real_type> w{};  // empty -> no need to befriend the model class!

                // TODO: OAA vs OAO
                // predict values using OAA -> num_data_points x num_classes
                plssvm::aos_matrix<plssvm::real_type> votes = self.call_predict_values(params, sv, alpha, rho, w, plssvm::bindings::python::util::pyarray_to_soa_matrix(predict_points));
                // votes *= plssvm::real_type{ -1.0 };  // TODO: sometimes necessary?
                return plssvm::bindings::python::util::matrix_to_pyarray(votes);
                // votes *= plssvm::real_type{ -1.0 };  // TODO: sometimes necessary? -> change label mapping in classification data set?!
            }, *self.model_); }, "Evaluate the decision function for the samples in X.")
        .def("fit", [](svc &self, py::object data, py::object labels, std::optional<std::vector<plssvm::real_type>> sample_weight) -> svc & {
            // sanity check parameter
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }

            // convert the labels to a std::vector
            const auto &[labels_vector_variant, dtype] = plssvm::bindings::python::util::pyobject_to_vector(labels);

            // convert the data py::object to a plssvm::aos_matrix
            const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);
            self.feature_names_ = opt_feature_names;

            // create the data set to fit
            std::visit([&](auto &&labels_vector) {
                // get the label type and possible data set types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                using possible_data_set_types = typename svc::possible_data_set_types;
                // create the data set to fit
                self.data_ = std::make_unique<possible_data_set_types>(plssvm::classification_data_set<label_type>(data_matrix, labels_vector));
            },
                       labels_vector_variant);

            // fit the model
            fit(self);
            return self; }, "Fit the SVM model according to the given training data.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt, py::return_value_policy::reference)
        .def("get_params", [](const svc &self, const bool) -> py::dict {
            const plssvm::parameter params = self.svm_->get_params();

            // fill a Python dictionary with the supported keys and values
            py::dict py_params;
            py_params["C"] = params.cost;
            py_params["break_ties"] = false;
            py_params["cache_size"] = 0;
            py_params["class_weight"] = py::none();
            py_params["coef0"] = params.coef0;
            py_params["decision_function_shape"] = self.classification == plssvm::classification_type::oaa ? "ovr" : "ovo";
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
            py_params["max_iter"] = self.max_iter.has_value() ? static_cast<long long>(self.max_iter.value()) : -1;
            py_params["probability"] = false;
            py_params["random_state"] = py::none();
            py_params["shrinking"] = false;
            py_params["tol"] = self.epsilon.value_or(plssvm::real_type{ 1e-10 });
            py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;

            return py_params; }, "Get parameters for this estimator.", py::arg("deep") = true)
        .def("predict", [](svc &self, py::object data) -> py::array {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            // convert the data py::object to a plssvm::aos_matrix
            const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);

            return std::visit([&](auto &&model) {
                // get the label type
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                // create the data set to predict
                const plssvm::classification_data_set<label_type> data_to_predict{ data_matrix };
                // predict the data
                return plssvm::bindings::python::util::vector_to_pyarray<label_type>(self.svm_->predict(model, data_to_predict));
            }, *self.model_); }, "Perform classification on samples in X.")
        .def("predict_log_proba", [](const svc &, py::array_t<plssvm::real_type>) { throw py::attribute_error{ "'SVC' object has no function 'predict_log_proba' (not implemented)" }; }, "Compute log probabilities of possible outcomes for samples in X.")
        .def("predict_proba", [](const svc &, py::array_t<plssvm::real_type>) { throw py::attribute_error{ "'SVC' object has no function 'predict_proba' (not implemented)" }; }, "Compute probabilities of possible outcomes for samples in X.")
        .def("score", [](svc &self, py::object data, py::object labels, std::optional<std::vector<plssvm::real_type>> sample_weight) -> plssvm::real_type {
            // sanity check parameter
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            }

            // convert the labels to a std::vector
            const auto &[labels_vector_variant, dtype] = plssvm::bindings::python::util::pyobject_to_vector(labels);

            // convert the data py::object to a plssvm::aos_matrix
            const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);

            // score the data
            return std::visit([&](auto &&labels_vector) {
                // get the label types
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                // create the data set to score
                const plssvm::classification_data_set<label_type> data_to_score{ data_matrix, labels_vector };
                // score the data
                try {
                    return self.svm_->score(std::get<plssvm::classification_model<label_type>>(*self.model_), data_to_score);
                } catch (const std::exception &) {
                    throw py::attribute_error{ fmt::format(R"(The dtype of the labels to score is "{}", but the model was fitted with "{}". Please use the same types for fit and score!)", dtype.attr("name").cast<std::string>(), self.py_dtype.attr("name").cast<std::string>()) };
                }
            }, labels_vector_variant); }, "Return the mean accuracy on the given test data and labels.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("set_params", [](svc &self, const py::kwargs &args) -> svc & {
            parse_provided_kwargs(self, args);
            return self; }, "Set the parameters of this estimator.", py::return_value_policy::reference)
        .def("__sklearn_is_fitted__", [](const svc &self) -> bool { return self.model_ != nullptr; }, "Return True if the estimator is fitted, False otherwise.")
        .def("__sklearn_clone__", [](const svc &self) -> svc {
            // create a new SVC instance
            svc new_svc{};
            // copy the parameters
            new_svc.svm_->set_params(self.svm_->get_params());
            new_svc.epsilon = self.epsilon;
            new_svc.max_iter = self.max_iter;
            new_svc.classification = self.classification;
            return new_svc; }, "Clone the estimator.");
}

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/core.hpp"

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, pyarray_t_to_vector, pyarray_to_matrix}

#include "fmt/format.h"          // fmt::format
#include "pybind11/numpy.h"      // support for STL types
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self, py::dynamic_attr
#include "pybind11/stl.h"        // support for STL types

#include <cstddef>   // std::size_t
#include <cstdint>   // std::int32_t
#include <map>       // std::map
#include <memory>    // std::unique_ptr, std::make_unique
#include <numeric>   // std::iota
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <utility>   // std::move
#include <variant>   // std::holds_alternative
#include <vector>    // std::vector

namespace py = pybind11;

// TODO: implement missing functionality (as far es possible)

// dummy
struct svr {
    // the types
    using real_type = plssvm::real_type;
    using label_type = plssvm::real_type;
    using data_set_type = plssvm::regression_data_set<label_type>;
    using model_type = plssvm::regression_model<label_type>;

    std::optional<real_type> epsilon{};
    std::optional<unsigned long long> max_iter{};

    std::unique_ptr<plssvm::csvr> svm_{ plssvm::make_csvr() };
    std::unique_ptr<data_set_type> data_{};
    std::unique_ptr<model_type> model_{};
};

void parse_provided_params(svr &self, const py::kwargs &args) {
    // check keyword arguments
    plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "tol", "cache_size", "verbose", "max_iter", "epsilon" });

    if (args.contains("C")) {
        self.svm_->set_params(plssvm::cost = args["C"].cast<typename svr::real_type>());
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
            throw py::attribute_error{ R"(The "kernel = 'precomputed'" parameter for the 'SVR' is not implemented yet!)" };
        } else {
            throw py::value_error{ fmt::format("'{}' is not in list", kernel_str) };
        }
        self.svm_->set_params(plssvm::kernel_type = kernel);
    } else {
        // sklearn default kernel is the rbf kernel
        self.svm_->set_params(plssvm::kernel_type = plssvm::kernel_function_type::rbf);
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
        self.svm_->set_params(plssvm::coef0 = args["coef0"].cast<typename svr::real_type>());
    }
    if (args.contains("shrinking")) {
        throw py::attribute_error{ "The 'shrinking' parameter for the 'SVR' is not implemented yet!" };
    }
    if (args.contains("tol")) {
        self.epsilon = args["tol"].cast<typename svr::real_type>();
    }
    if (args.contains("cache_size")) {
        throw py::attribute_error{ "The 'cache_size' parameter for the 'SVR' is not implemented yet!" };
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
    } else {
        // sklearn default is quiet
        plssvm::verbosity = plssvm::verbosity_level::quiet;
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
    if (args.contains("epsilon")) {
        throw py::attribute_error{ "The 'epsilon' parameter for the 'SVR' is not implemented yet!" };
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
    if (self.epsilon.has_value() && self.epsilon.value() <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "eps <= 0" };
    }

    // fit the model using potentially provided keyword arguments
    if (self.epsilon.has_value() && self.max_iter.has_value()) {
        self.model_ = std::make_unique<typename svr::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::epsilon = self.epsilon.value(),
                                                                                plssvm::max_iter = self.max_iter.value()));
    } else if (self.epsilon.has_value()) {
        self.model_ = std::make_unique<typename svr::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::epsilon = self.epsilon.value()));
    } else if (self.max_iter.has_value()) {
        self.model_ = std::make_unique<typename svr::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::max_iter = self.max_iter.value()));
    } else {
        self.model_ = std::make_unique<typename svr::model_type>(self.svm_->fit(*self.data_));
    }
}

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
                   parse_provided_params(*self, args);
                   return self;
               }),
               "Construct a new SVM classifier.");

    //*************************************************************************************************************************************//
    //                                                             ATTRIBUTES                                                              //
    //*************************************************************************************************************************************//
    py_svr.def_property_readonly("coef_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'coef_' (not implemented)" }; })
        .def_property_readonly("dual_coef_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'dual_coef_' (not implemented)" }; })
        .def_property_readonly("fit_status_", [](const svr &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVR' object has no attribute 'fit_status_'" };
                } else {
                    return 0;
                } }, "0 if correctly fitted, 1 otherwise (will raise exception). int")
        .def_property_readonly("intercept_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'intercept_' (not implemented)" }; })
        .def_property_readonly("n_features_in_", [](const svr &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVR' object has no attribute 'n_features_in_'" };
                } else {
                    return static_cast<int>(self.data_->num_features());
                } }, "Number of features seen during fit. int")
        .def_property_readonly("feature_names_in_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'feature_names_in_' (not implemented)" }; })
        .def_property_readonly("n_iter_", [](const svr &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVR' object has no attribute 'support_'" };
            } else {
                return plssvm::bindings::python::util::vector_to_pyarray(self.model_->num_iters().value());
            } })
        .def_property_readonly("support_", [](const svr &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVR' object has no attribute 'support_'" };
                } else {
                    // for the SVR, the indices do not need to be sorted
                    std::vector<int> support(self.model_->num_support_vectors());
                    std::iota(support.begin(), support.end(), 0);
                    return plssvm::bindings::python::util::vector_to_pyarray(support);
                } }, "Indices of support vectors. ndarray of shape (n_SV)")
        .def_property_readonly("support_vectors_", [](const svr &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVR' object has no attribute 'support_vectors_'" };
                } else {
                    // for the SVR, the support vectors do not need to be sorted
                    // convert 2D vector back to plssvm::matrix
                    return plssvm::bindings::python::util::matrix_to_pyarray(plssvm::aos_matrix<plssvm::real_type>{ std::move(self.model_->support_vectors().to_2D_vector()) });
                } }, "Support vectors. ndarray of shape (n_SV, n_features)")
        .def_property_readonly("n_support_", [](const svr &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVR' object has no attribute 'n_support_'" };
                } else {
                    // for SVR, only report the total number of support vectors
                    return plssvm::bindings::python::util::vector_to_pyarray(std::vector<std::int32_t>{ static_cast<std::int32_t>(self.model_->num_support_vectors()) });
                } }, "Number of support vectors for each class. ndarray of shape (1,), dtype=int32")
        .def_property_readonly("probA_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'probA_' (not implemented)" }; })
        .def_property_readonly("probB_", [](const svr &) { throw py::attribute_error{ "'SVR' object has no attribute 'probB_' (not implemented)" }; })
        .def_property_readonly("shape_fit_", [](const svr &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVR' object has no attribute 'shape_fit_'" };
                } else {
                    return std::make_tuple(static_cast<int>(self.data_->num_data_points()), static_cast<int>(self.data_->num_features()));
                } }, "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)");

    //*************************************************************************************************************************************//
    //                                                               METHODS                                                               //
    //*************************************************************************************************************************************//
    py_svr.def(
              "fit", [](svr &self, py::array_t<typename svr::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<typename svr::real_type, py::array::c_style | py::array::forcecast> labels, const std::optional<std::vector<typename svr::real_type>> &sample_weight) -> svr & {
                  if (sample_weight.has_value()) {
                      throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                  }

                  // fit the model using potentially provided keyword arguments
                  self.data_ = std::make_unique<typename svr::data_set_type>(plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pyarray_t_to_vector(labels));
                  fit(self);
                  return self;
              },
              "Fit the SVM model according to the given training data.",
              py::arg("X"),
              py::arg("y"),
              py::pos_only(),
              py::arg("sample_weight") = std::nullopt,
              py::return_value_policy::reference)
        .def("get_params", [](const svr &self, const bool) {
                  const plssvm::parameter params = self.svm_->get_params();

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
                  py_params["max_iter"] = self.max_iter.has_value() ? static_cast<long long>(self.max_iter.value()) : -1;
                  py_params["shrinking"] = false;
                  py_params["tol"] = self.epsilon.value_or(typename svr::real_type{ 1e-3 });
                  py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;

                  return py_params; }, "Get parameters for this estimator.", py::arg("deep") = true)
        .def("predict", [](svr &self, py::array_t<typename svr::real_type, py::array::c_style | py::array::forcecast> data) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                } else {
                    const typename svr::data_set_type data_to_predict{ plssvm::bindings::python::util::pyarray_to_matrix(data) };
                    return plssvm::bindings::python::util::vector_to_pyarray(self.svm_->predict(*self.model_, data_to_predict));
                } }, "Perform classification on samples in X.")
        .def("score", [](svr &self, py::array_t<typename svr::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<typename svr::real_type, py::array::c_style | py::array::forcecast> labels, const std::optional<std::vector<typename svr::real_type>> &sample_weight) {
                  if (sample_weight.has_value()) {
                      throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                  }

                  if (self.model_ == nullptr) {
                      throw py::attribute_error{ "This SVR instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                  } else {
                      const typename svr::data_set_type data_to_score{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pyarray_t_to_vector(labels) };
                      return self.svm_->score(*self.model_, data_to_score);
                  } }, "Return the mean accuracy on the given test data and labels.", py::arg("X"), py::arg("y"), py::pos_only(), py::arg("sample_weight") = std::nullopt)
        .def("set_params", [](svr &self, const py::kwargs &args) -> svr & {
            parse_provided_params(self, args);
            return self; }, "Set the parameters of this estimator.", py::return_value_policy::reference)
        .def("__sklearn_is_fitted__", [](const svr &self) { return self.model_ != nullptr; })
        .def("__sklearn_clone__", [](const svr &self) {
            // create a new SVR instance
            svr new_svr{};
            // copy the parameters
            new_svr.svm_->set_params(self.svm_->get_params());
            new_svr.epsilon = self.epsilon;
            new_svr.max_iter = self.max_iter;
            return new_svr; });
}

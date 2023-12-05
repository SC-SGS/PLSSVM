/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/core.hpp"

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "utility.hpp"  // check_kwargs_for_correctness, assemble_unique_class_name, pyarray_to_vector, pyarray_to_matrix

#include "fmt/core.h"            // fmt::format
#include "pybind11/numpy.h"      // support for STL types
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self
#include "pybind11/stl.h"        // support for STL types

#include <cstddef>   // std::size_t
#include <cstdint>   // std::int32_t
#include <map>       // std::map
#include <memory>    // std::unique_ptr, std::make_unique
#include <numeric>   // std::iota
#include <optional>  // std::optional, std::nullopt
#include <sstream>   // std::stringstream
#include <string>    // std::string
#include <tuple>     // std::tuple
#include <vector>    // std::vector

namespace py = pybind11;

// TODO: implement missing functionality
// TODO: fit -> return self

// dummy
struct svc {
    // the types
    using real_type = plssvm::real_type;
    using label_type = PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE;
    using data_set_type = plssvm::data_set<label_type>;
    using model_type = plssvm::model<label_type>;

    std::optional<real_type> epsilon{};
    std::optional<unsigned long long> max_iter{};
    plssvm::classification_type classification{ plssvm::classification_type::oaa };

    std::unique_ptr<plssvm::csvm> svm_{ plssvm::make_csvm() };
    std::unique_ptr<data_set_type> data_{};
    std::unique_ptr<model_type> model_{};
};

void parse_provided_params(svc &self, const py::kwargs &args) {
    // check keyword arguments
    check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size", "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state", "classification" });

    if (args.contains("C")) {
        self.svm_->set_params(plssvm::cost = args["C"].cast<typename svc::real_type>());
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
        } else if (kernel_str == "sigmoid" || kernel_str == "precomputed") {
            throw py::attribute_error{ R"(The "kernel = 'sigmoid'" or "kernel = 'precomputed'" parameter for a call to the 'SVC' constructor is not implemented yet!)" };
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
        if (py::isinstance<py::str>(args["gamma"])) {
            // found a string
            const auto gamma = args["gamma"].cast<std::string>();
            if (gamma == "scale") {
                // TODO: implement sklearn's scale option?
                throw py::attribute_error{ "The \"gamma = 'scale'\" parameter for a call to the 'SVC' constructor is not implemented yet!" };
            } else if (gamma == "auto") {
                // default behavior in PLSSVM -> do nothing
            } else {
                throw py::value_error{ fmt::format("When 'gamma' is a string, it should be either 'scale' or 'auto'. Got '{}' instead.", gamma) };
            }
        } else {
            const auto gamma = args["gamma"].cast<plssvm::real_type>();
            if (gamma <= plssvm::real_type{ 0.0 }) {
                throw py::value_error{ fmt::format("gamma value must be > 0; {} is invalid. Use a positive number or use 'auto' to set gamma to a value of 1 / n_features.", gamma) };
            }
            self.svm_->set_params(plssvm::gamma = gamma);
        }
    }
    if (args.contains("coef0")) {
        self.svm_->set_params(plssvm::coef0 = args["coef0"].cast<typename svc::real_type>());
    }
    if (args.contains("shrinking")) {
        throw py::attribute_error{ "The 'shrinking' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
    if (args.contains("probability")) {
        throw py::attribute_error{ "The 'probability' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
    if (args.contains("tol")) {
        self.epsilon = args["tol"].cast<typename svc::real_type>();
    }
    if (args.contains("cache_size")) {
        throw py::attribute_error{ "The 'cache_size' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
    if (args.contains("class_weight")) {
        throw py::attribute_error{ "The 'class_weight' parameter for a call to the 'SVC' constructor is not implemented yet!" };
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
        throw py::attribute_error{ "The 'break_ties' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
    if (args.contains("random_state")) {
        throw py::attribute_error{ "The 'random_state' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
}

void fit(svc &self) {
    // perform sanity checks
    if (self.svm_->get_params().cost.value() <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "C <= 0" };
    }
    if (self.svm_->get_params().degree.value() < 0) {
        throw py::value_error{ "degree of polynomial kernel < 0" };
    }
    if (self.epsilon.has_value() && self.epsilon.value() <= plssvm::real_type{ 0.0 }) {
        throw py::value_error{ "eps <= 0" };
    }

    // fit the model using potentially provided keyword arguments
    if (self.epsilon.has_value() && self.max_iter.has_value()) {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = self.classification,
                                                                                plssvm::epsilon = self.epsilon.value(),
                                                                                plssvm::max_iter = self.max_iter.value()));
    } else if (self.epsilon.has_value()) {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = self.classification,
                                                                                plssvm::epsilon = self.epsilon.value()));
    } else if (self.max_iter.has_value()) {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = self.classification,
                                                                                plssvm::max_iter = self.max_iter.value()));
    } else {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = self.classification));
    }
}

template <typename svc>
[[nodiscard]] std::vector<int> calculate_sv_indices_per_class(const svc &self) {
    std::map<typename svc::label_type, std::vector<int>> indices_per_class{};
    // init index-map map
    for (const typename svc::label_type &label : self.model_->classes()) {
        indices_per_class.insert({ label, std::vector<int>{} });
    }
    // sort the indices into the respective bucket based on their associated class
    for (std::size_t idx = 0; idx < self.model_->num_support_vectors(); ++idx) {
        indices_per_class[self.model_->labels()[idx]].push_back(static_cast<int>(idx));
    }
    // convert map values to vector
    std::vector<int> support{};
    support.reserve(self.model_->num_support_vectors());
    for (const auto &[label, indices] : indices_per_class) {
        support.insert(support.cend(), indices.cbegin(), indices.cend());
    }
    return support;
}

void init_sklearn(py::module_ &m) {
    // documentation based on sklearn.svm.SVC documentation
    py::class_<svc> py_svc(m, "SVC");
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
                   parse_provided_params(*self, args);
                   return self;
               }),
               "Construct a new SVM classifier.");

    //*************************************************************************************************************************************//
    //                                                             ATTRIBUTES                                                              //
    //*************************************************************************************************************************************//
    py_svc.def_property_readonly("class_weight_", [](const svc &self) {
              if (self.model_ == nullptr) {
                  throw py::attribute_error{ "'SVC' object has no attribute 'class_weight_'" };
              } else {
                  // note: constant zero since the class_weight parameter is currently not supported
                  const auto size = static_cast<int>(self.model_->num_classes());
                  py::array_t<plssvm::real_type, py::array::c_style> py_array(size);
                  const py::buffer_info buffer = py_array.request();
                  auto ptr = static_cast<plssvm::real_type *>(buffer.ptr);
                  std::fill(ptr, ptr + size, plssvm::real_type{ 1.0 });
                  return py_array;
              }
          })
        .def_property_readonly(
            "classes_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'classes_'" };
                } else {
                    return vector_to_pyarray(self.data_->classes().value());
                }
            },
            "The classes labels. ndarray of shape (n_classes,)")
        .def_property_readonly("coef_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'coef_' (not implemented)" };
        })
        .def_property_readonly("dual_coef_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'dual_coef_' (not implemented)" };
        })
        .def_property_readonly(
            "fit_status_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'fit_status_'" };
                } else {
                    return 0;
                }
            },
            "0 if correctly fitted, 1 otherwise (will raise exception). int")
        .def_property_readonly("intercept_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'intercept_' (not implemented)" };
        })
        .def_property_readonly(
            "n_features_in_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'n_features_in_'" };
                } else {
                    return static_cast<int>(self.data_->num_features());
                }
            },
            "Number of features seen during fit. int")
        .def_property_readonly("feature_names_in_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'feature_names_in_' (not implemented)" };
        })
        .def_property_readonly("n_iter_", [](const svc &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
            } else {
                return vector_to_pyarray(self.model_->num_iters().value());
            }
        })
        .def_property_readonly(
            "support_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
                } else {
                    return vector_to_pyarray(calculate_sv_indices_per_class(self));
                }
            },
            "Indices of support vectors. ndarray of shape (n_SV)")
        .def_property_readonly(
            "support_vectors_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'support_vectors_'" };
                } else {
                    // get the sorted indices
                    const std::vector<int> support = calculate_sv_indices_per_class(self);
                    // convert support vectors matrix to 2d vector
                    std::vector<std::vector<plssvm::real_type>> sv = self.model_->support_vectors().to_2D_vector();

                    // sort support vectors by their class
                    std::vector<std::vector<plssvm::real_type>> sorted_sv{};
                    sorted_sv.reserve(sv.size());
                    for (const int idx : support) {
                        sorted_sv.push_back(std::move(sv[idx]));
                    }

                    // convert 2D vector back to plssvm::matrix
                    return matrix_to_pyarray(plssvm::aos_matrix<plssvm::real_type>{ std::move(sorted_sv) });
                }
            },
            "Support vectors. ndarray of shape (n_SV, n_features)")
        .def_property_readonly(
            "n_support_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'n_support_'" };
                } else {
                    std::map<typename svc::label_type, std::int32_t> occurrences{};
                    // init count map
                    for (const typename svc::label_type &label : self.model_->classes()) {
                        occurrences.insert({ label, std::int32_t{ 0 } });
                    }
                    // count occurrences
                    for (const typename svc::label_type &label : self.model_->labels()) {
                        ++occurrences[label];
                    }
                    // convert map values to vector
                    std::vector<std::int32_t> n_support{};
                    n_support.reserve(occurrences.size());
                    for (const auto &[label, n_sv] : occurrences) {
                        n_support.push_back(n_sv);
                    }
                    // convert to Numpy array
                    return vector_to_pyarray(n_support);
                }
            },
            "Number of support vectors for each class. ndarray of shape (n_classes,), dtype=int32")
        .def_property_readonly("probA_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'probA_' (not implemented)" };
        })
        .def_property_readonly("probB_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'probB_' (not implemented)" };
        })
        .def_property_readonly(
            "shape_fit_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'shape_fit_'" };
                } else {
                    return std::make_tuple(static_cast<int>(self.data_->num_data_points()), static_cast<int>(self.data_->num_features()));
                }
            },
            "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)");

    //*************************************************************************************************************************************//
    //                                                               METHODS                                                               //
    //*************************************************************************************************************************************//
    py_svc.def("decision_function", [](const svc &, py::array_t<typename svc::real_type>) {
        // TODO: predict_values?!
        throw py::attribute_error{ "'SVC' object has no function 'decision_function' (not implemented)" };
    });
#if !defined(PLSSVM_PYTHON_BINDINGS_LABEL_TYPE_IS_STRING)
    py_svc.def(
        "fit", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<typename svc::label_type, py::array::c_style | py::array::forcecast> labels, std::optional<std::vector<typename svc::real_type>> sample_weight) -> svc & {
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }

            // fit the model using potentially provided keyword arguments
            self.data_ = std::make_unique<typename svc::data_set_type>(pyarray_to_matrix(data), pyarray_to_vector(labels));
            fit(self);
            return self;
        },
        "Fit the SVM model according to the given training data.",
        py::arg("X"),
        py::arg("y"),
        py::pos_only(),
        py::arg("sample_weight") = std::nullopt,
        py::return_value_policy::reference);
#else
    py_svc.def(
              "fit", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) -> svc & {
                  if (sample_weight.has_value()) {
                      throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                  }

                  // fit the model using potentially provided keyword arguments
                  self.data_ = std::make_unique<typename svc::data_set_type>(pyarray_to_matrix(data), pyarray_to_string_vector(labels));
                  fit(self);
                  return self;
              },
              "Fit the SVM model according to the given training data.",
              py::arg("X"),
              py::arg("y"),
              py::pos_only(),
              py::arg("sample_weight") = std::nullopt,
              py::return_value_policy::reference)
        .def(
            "fit", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data, const py::list &labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) -> svc & {
                if (sample_weight.has_value()) {
                    throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                }

                // fit the model using potentially provided keyword arguments
                self.data_ = std::make_unique<typename svc::data_set_type>(pyarray_to_matrix(data), pylist_to_string_vector(labels));
                fit(self);
                return self;
            },
            "Fit the SVM model according to the given training data.",
            py::arg("X"),
            py::arg("y"),
            py::pos_only(),
            py::arg("sample_weight") = std::nullopt,
            py::return_value_policy::reference);
#endif
    py_svc.def(
              "get_params", [](const svc &self, const bool) {
                  const plssvm::parameter params = self.svm_->get_params();

                  // fill a Python dictionary with the supported keys and values
                  py::dict py_params;
                  py_params["C"] = params.cost.value();
                  py_params["break_ties"] = false;
                  py_params["cache_size"] = 0;
                  py_params["class_weight"] = py::none();
                  py_params["coef0"] = params.coef0.value();
                  py_params["decision_function_shape"] = self.classification == plssvm::classification_type::oaa ? "ovr" : "ovo";
                  py_params["degree"] = params.degree.value();
                  // TODO: implemented sklearn gamma = 'scale'
                  if (params.gamma.is_default()) {
                      py_params["gamma"] = "auto";
                  } else {
                      py_params["gamma"] = params.gamma.value();
                  }
                  py_params["kernel"] = fmt::format("{}", params.kernel_type);
                  py_params["max_iter"] = self.max_iter.has_value() ? static_cast<long long>(self.max_iter.value()) : -1;
                  py_params["probability"] = false;
                  py_params["random_state"] = py::none();
                  py_params["shrinking"] = false;
                  py_params["tol"] = self.epsilon.value_or(typename svc::real_type{ 1e-3 });
                  py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;

                  return py_params;
              },
              "Get parameters for this estimator.",
              py::arg("depp") = true)
        .def(
            "predict", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                } else {
                    const typename svc::data_set_type data_to_predict{ pyarray_to_matrix(data) };
                    return vector_to_pyarray(self.svm_->predict(*self.model_, data_to_predict));
                }
            },
            "Perform classification on samples in X.")
        .def("predict_log_proba", [](const svc &, py::array_t<typename svc::real_type>) {
            throw py::attribute_error{ "'SVC' object has no function 'predict_log_proba' (not implemented)" };
        })
        .def("predict_proba", [](const svc &, py::array_t<typename svc::real_type>) {
            throw py::attribute_error{ "'SVC' object has no function 'predict_proba' (not implemented)" };
        });
#if !defined(PLSSVM_PYTHON_BINDINGS_LABEL_TYPE_IS_STRING)
    py_svc.def(
        "score", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<typename svc::label_type, py::array::c_style | py::array::forcecast> labels, std::optional<std::vector<typename svc::real_type>> sample_weight) {
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }

            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            } else {
                const typename svc::data_set_type data_to_score{ pyarray_to_matrix(data), pyarray_to_vector(labels) };
                return self.svm_->score(*self.model_, data_to_score);
            }
        },
        "Return the mean accuracy on the given test data and labels.",
        py::arg("X"),
        py::arg("y"),
        py::pos_only(),
        py::arg("sample_weight") = std::nullopt);
#else
    py_svc.def(
              "score", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) {
                  if (sample_weight.has_value()) {
                      throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                  }

                  if (self.model_ == nullptr) {
                      throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                  } else {
                      const typename svc::data_set_type data_to_score{ pyarray_to_matrix(data), pyarray_to_string_vector(labels) };
                      return self.svm_->score(*self.model_, data_to_score);
                  }
              },
              "Return the mean accuracy on the given test data and labels.",
              py::arg("X"),
              py::arg("y"),
              py::pos_only(),
              py::arg("sample_weight") = std::nullopt)
        .def(
            "score", [](svc &self, py::array_t<typename svc::real_type, py::array::c_style | py::array::forcecast> data, py::list labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) {
                if (sample_weight.has_value()) {
                    throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                }

                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                } else {
                    const typename svc::data_set_type data_to_score{ pyarray_to_matrix(data), pylist_to_string_vector(labels) };
                    return self.svm_->score(*self.model_, data_to_score);
                }
            },
            "Return the mean accuracy on the given test data and labels.",
            py::arg("X"),
            py::arg("y"),
            py::pos_only(),
            py::arg("sample_weight") = std::nullopt);
#endif
    py_svc.def(
        "set_params", [](svc &self, const py::kwargs &args) -> svc & {
            parse_provided_params(self, args);
            return self;
        },
        "Set the parameters of this estimator.",
        py::return_value_policy::reference);
}
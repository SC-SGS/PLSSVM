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

// dummy
struct svc {
    // the types
    using real_type = plssvm::real_type;
    using label_type = PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE;
    using data_set_type = plssvm::data_set<label_type>;
    using model_type = plssvm::model<label_type>;

    std::optional<real_type> epsilon{};
    std::optional<long long> max_iter{};

    std::unique_ptr<plssvm::csvm> svm_{ plssvm::make_csvm() };
    std::unique_ptr<data_set_type> data_{};
    std::unique_ptr<model_type> model_{};
};

void parse_provided_params(svc &self, const py::kwargs &args) {
    // check keyword arguments
    check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size", "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state" });

    if (args.contains("C")) {
        self.svm_->set_params(plssvm::cost = args["C"].cast<typename svc::real_type>());
    }
    if (args.contains("kernel")) {
        std::stringstream ss{ args["kernel"].cast<std::string>() };
        plssvm::kernel_function_type kernel{};
        ss >> kernel;
        self.svm_->set_params(plssvm::kernel_type = kernel);
    }
    if (args.contains("degree")) {
        self.svm_->set_params(plssvm::degree = args["degree"].cast<int>());
    }
    if (args.contains("gamma")) {
        // TODO: correctly reflect sklearn's scale, auto, and value options
        self.svm_->set_params(plssvm::gamma = args["gamma"].cast<typename svc::real_type>());
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
            plssvm::verbosity = plssvm::verbosity_level::full;
        } else {
            plssvm::verbosity = plssvm::verbosity_level::quiet;
        }
    }
    if (args.contains("max_iter")) {
        self.max_iter = args["max_iter"].cast<long long>();
    }
    if (args.contains("decision_function_shape")) {
        throw py::attribute_error{ "The 'decision_function_shape' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
    if (args.contains("break_ties")) {
        throw py::attribute_error{ "The 'break_ties' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
    if (args.contains("random_state")) {
        throw py::attribute_error{ "The 'random_state' parameter for a call to the 'SVC' constructor is not implemented yet!" };
    }
}

void fit(svc &self) {
    // fit the model using potentially provided keyword arguments
    if (self.epsilon.has_value() && self.max_iter.has_value()) {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = plssvm::classification_type::oao,
                                                                                plssvm::epsilon = self.epsilon.value(),
                                                                                plssvm::max_iter = self.max_iter.value()));
    } else if (self.epsilon.has_value()) {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = plssvm::classification_type::oao,
                                                                                plssvm::epsilon = self.epsilon.value()));
    } else if (self.max_iter.has_value()) {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = plssvm::classification_type::oao,
                                                                                plssvm::max_iter = self.max_iter.value()));
    } else {
        self.model_ = std::make_unique<typename svc::model_type>(self.svm_->fit(*self.data_,
                                                                                plssvm::classification = plssvm::classification_type::oao));
    }
}

void init_sklearn(py::module_ &m) {
    // documentation based on sklearn.svm.SVC documentation
    py::class_<svc> py_svc(m, "SVC");
    py_svc.def(py::init([](const py::kwargs &args) {
                   // to silence constructor messages
                   if (args.contains("verbose")) {
                       if (args["verbose"].cast<bool>()) {
                           plssvm::verbosity = plssvm::verbosity_level::full;
                       } else {
                           plssvm::verbosity = plssvm::verbosity_level::quiet;
                       }
                   }

                   // create SVC class
                   auto self = std::make_unique<svc>();
                   parse_provided_params(*self, args);
                   return self;
               }),
               "Construct a new SVM classifier.")

        // FUNCTIONS
        .def("decision_function", [](const svc &, py::array_t<typename svc::real_type>) {
            throw py::attribute_error{ "'SVC' object has no function 'decision_function' (not implemented)" };
        });
#if !defined(PLSSVM_PYTHON_BINDINGS_LABEL_TYPE_IS_STRING)
    py_svc.def(
        "fit", [](svc &self, py::array_t<typename svc::real_type> data, py::array_t<typename svc::label_type> labels, std::optional<std::vector<typename svc::real_type>> sample_weight) {
            if (sample_weight.has_value()) {
                throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
            }

            // fit the model using potentially provided keyword arguments
            self.data_ = std::make_unique<typename svc::data_set_type>(pyarray_to_matrix(data), pyarray_to_vector(labels));
            fit(self);
        },
        "Fit the SVM model according to the given training data.",
        py::arg("X"),
        py::arg("y"),
        py::pos_only(),
        py::arg("sample_weight") = std::nullopt);
#else
    py_svc.def(
              "fit", [](svc &self, py::array_t<typename svc::real_type> data, py::array_t<typename svc::real_type> labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) {
                  if (sample_weight.has_value()) {
                      throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                  }

                  // fit the model using potentially provided keyword arguments
                  self.data_ = std::make_unique<typename svc::data_set_type>(pyarray_to_matrix(data), pyarray_to_string_vector(labels));
                  fit(self);
              },
              "Fit the SVM model according to the given training data.",
              py::arg("X"),
              py::arg("y"),
              py::pos_only(),
              py::arg("sample_weight") = std::nullopt)
        .def(
            "fit", [](svc &self, py::array_t<typename svc::real_type> data, const py::list &labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) {
                if (sample_weight.has_value()) {
                    throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                }

                // fit the model using potentially provided keyword arguments
                self.data_ = std::make_unique<typename svc::data_set_type>(pyarray_to_matrix(data), pylist_to_string_vector(labels));
                fit(self);
            },
            "Fit the SVM model according to the given training data.",
            py::arg("X"),
            py::arg("y"),
            py::pos_only(),
            py::arg("sample_weight") = std::nullopt);
#endif
    py_svc.def(
              "get_params", [](const svc &self) {
                  const plssvm::parameter params = self.svm_->get_params();

                  // fill a Python dictionary with the supported keys and values
                  py::dict py_params;
                  py_params["C"] = params.cost.value();
                  py_params["kernel"] = fmt::format("{}", params.kernel_type);
                  py_params["degree"] = params.degree.value();
                  // TODO: correctly reflect sklearn's scale, auto, and value options
                  py_params["gamma"] = params.gamma.value();
                  py_params["coef0"] = params.coef0.value();
                  // py_params["shrinking"];
                  // py_params["probability"];
                  py_params["tol"] = self.epsilon.value_or(typename svc::real_type{ 1e-3 });
                  // py_params["cache_size"];
                  // py_params["class_weight"];
                  py_params["verbose"] = plssvm::verbosity != plssvm::verbosity_level::quiet;
                  py_params["max_iter"] = self.max_iter.value_or(-1);
                  // py_params["decision_function_shape"];
                  // py_params["break_ties"];
                  // py_params["random_state"];

                  return py_params;
              },
              "Get parameters for this estimator.")
        .def(
            "predict", [](svc &self, py::array_t<typename svc::real_type> data) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                } else {
                    const typename svc::data_set_type data_to_predict{ pyarray_to_matrix(data) };
                    if constexpr (std::is_same_v<typename svc::label_type, std::string>) {
                        return self.svm_->predict(*self.model_, data_to_predict);
                    } else {
                        return vector_to_pyarray(self.svm_->predict(*self.model_, data_to_predict));
                    }
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
        "score", [](svc &self, py::array_t<typename svc::real_type> data, py::array_t<typename svc::label_type> labels, std::optional<std::vector<typename svc::real_type>> sample_weight) {
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
              "score", [](svc &self, py::array_t<typename svc::real_type> data, py::array_t<typename svc::real_type> labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) {
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
            "score", [](svc &self, py::array_t<typename svc::real_type> data, py::list labels, const std::optional<std::vector<typename svc::real_type>> &sample_weight) {
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
              "set_params", [](svc &self, const py::kwargs &args) {
                  parse_provided_params(self, args);
              },
              "Set the parameters of this estimator.")

        // ATTRIBUTES
        .def_property_readonly("class_weight_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'class_weight_' (not implemented)" };
        })
        .def_property_readonly(
            "classes_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'classes_'" };
                } else {
                    if constexpr (std::is_same_v<typename svc::label_type, std::string>) {
                        return self.data_->classes().value();
                    } else {
                        return vector_to_pyarray(self.data_->classes().value());
                    }
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
        .def_property_readonly("n_iter_", [](const svc &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'n_iter_' (not implemented)" };
        })
        .def_property_readonly(
            "support_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
                } else {
                    // all data points are support vectors
                    const auto size = static_cast<int>(self.model_->num_support_vectors());
                    py::array_t<int> py_array(size);
                    const py::buffer_info buffer = py_array.request();
                    int *ptr = static_cast<int *>(buffer.ptr);
                    for (int i = 0; i < size; ++i) {
                        ptr[i] = i;
                    }
                    return py_array;
                }
            },
            "Indices of support vectors. ndarray of shape (n_SV)")
        .def_property_readonly(
            "support_vectors_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'support_vectors_'" };
                } else {
                    // all data points are support vectors
                    return matrix_to_pyarray(self.model_->support_vectors());
                }
            },
            "Support vectors. ndarray of shape (n_SV, n_features)")
        .def_property_readonly(
            "n_support_", [](const svc &self) {
                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "'SVC' object has no attribute 'n_support_'" };
                } else {
                    // TODO: correct implementation?
                    // all data points are support vectors
                    std::vector<std::int32_t> n_support(self.model_->num_classes(), self.model_->num_support_vectors());
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
                    return std::tuple<int, int>{ static_cast<int>(self.data_->num_data_points()), static_cast<int>(self.data_->num_features()) };
                }
            },
            "Array dimensions of training vector X. tuple of int of shape (n_dimensions_of_X,)");
}
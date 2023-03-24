#include "plssvm/core.hpp"

#include "utility.hpp"

#include "fmt/core.h"            // fmt::format
#include "pybind11/numpy.h"      // support for STL types
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self
#include "pybind11/stl.h"        // support for STL types

#include <sstream>

namespace py = pybind11;

// TODO: no copy-paste
template <typename T>
std::vector<std::vector<T>> pyarray_to_vector_of_vector(py::array_t<T> data) {
    // check dimensions
    if (data.ndim() != 2) {
        throw py::value_error{ fmt::format("the provided array must have exactly two dimensions but has {}!", data.ndim()) };
    }

    // convert py::array to std::vector<std::vector<>>
    std::vector<std::vector<T>> tmp(data.shape(0));
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        tmp[i] = std::vector<T>(data.data(i, 0), data.data(i, 0) + data.shape(1));
    }

    return tmp;
}

template <typename T>
std::vector<T> pyarray_to_vector(py::array_t<T> data) {
    // check dimensions
    if (data.ndim() != 1) {
        throw py::value_error{ fmt::format("the provided array must have exactly one dimension but has {}!", data.ndim()) };
    }

    // convert py::array to std::vector
    return std::vector<T>(data.data(0), data.data(0) + data.shape(0));
}

// dummy
template <typename T, typename U>
struct svc {
    using real_type = T;
    using label_type = U;
    using data_set_type = plssvm::data_set<real_type, label_type>;
    using model_type = plssvm::model<real_type, label_type>;

    //    template <typename... Args>
    //    svc(Args &&...args) :
    //        svm_{ plssvm::make_csvm(std::forward<Args>(args)...) } {}

    std::optional<real_type> epsilon{};
    std::optional<long long> max_iter{};

    std::unique_ptr<plssvm::csvm> svm_{ plssvm::make_csvm() };
    std::unique_ptr<data_set_type> data_{};
    std::unique_ptr<model_type> model_{};
};

void init_sklearn(py::module_ &m) {
    using svc_type = svc<PLSSVM_PYTHON_BINDINGS_PREFERRED_REAL_TYPE, PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>;

    py::class_<svc_type>(m, "SVC")

        .def(py::init([](py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size", "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state" });

            // to silence constructor messages
            if (args.contains("verbose")) {
                // TODO: doesn't currently not work!
                plssvm::verbose = args["verbose"].cast<bool>();
            }

            auto self = std::make_unique<svc_type>();

            if (args.contains("C")) {
                self->svm_->set_params(plssvm::cost = args["C"].cast<typename svc_type::real_type>());
            }
            if (args.contains("kernel")) {
                std::stringstream ss{ args["kernel"].cast<std::string>() };
                plssvm::kernel_function_type kernel;
                ss >> kernel;
                self->svm_->set_params(plssvm::kernel_type = kernel);
            }
            if (args.contains("degree")) {
                self->svm_->set_params(plssvm::degree = args["degree"].cast<int>());
            }
            if (args.contains("gamma")) {
                self->svm_->set_params(plssvm::gamma = args["gamma"].cast<typename svc_type::real_type>());  // TODO: scale, auto, value
            }
            if (args.contains("coef0")) {
                self->svm_->set_params(plssvm::coef0 = args["coef0"].cast<typename svc_type::real_type>());
            }
            if (args.contains("shrinking")) {
                throw py::attribute_error{ "The 'shrinking' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("probability")) {
                throw py::attribute_error{ "The 'probability' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("tol")) {
                self->epsilon.value() = args["tol"].cast<typename svc_type::real_type>();
            }
            if (args.contains("cache_size")) {
                throw py::attribute_error{ "The 'cache_size' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("class_weight")) {
                throw py::attribute_error{ "The 'class_weight' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("max_iter")) {
                self->max_iter.value() = args["max_iter"].cast<long long>();
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

            return self;
        }))
        .def("decision_function", [](const svc_type &, py::array_t<typename svc_type::real_type>) {
            throw py::attribute_error{ "'SVC' object has no function 'decision_function' (not implemented)" };
        })
        .def(
            "fit", [](svc_type &self, py::array_t<typename svc_type::real_type> data, py::array_t<typename svc_type::label_type> labels, std::optional<std::vector<typename svc_type::real_type>> sample_weight) {
                if (sample_weight.has_value()) {
                    throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                }

                self.data_ = std::make_unique<typename svc_type::data_set_type>(pyarray_to_vector_of_vector(data), pyarray_to_vector(labels));
                if (self.epsilon.has_value() && self.max_iter.has_value()) {
                    self.model_ = std::make_unique<typename svc_type::model_type>(self.svm_->fit(*self.data_, plssvm::epsilon = self.epsilon.value(), plssvm::max_iter = self.max_iter.value()));
                } else if (self.epsilon.has_value()) {
                    self.model_ = std::make_unique<typename svc_type::model_type>(self.svm_->fit(*self.data_, plssvm::epsilon = self.epsilon.value()));
                } else if (self.max_iter.has_value()) {
                    self.model_ = std::make_unique<typename svc_type::model_type>(self.svm_->fit(*self.data_, plssvm::max_iter = self.max_iter.value()));
                } else {
                    self.model_ = std::make_unique<typename svc_type::model_type>(self.svm_->fit(*self.data_));
                }
            },
            py::arg("X"),
            py::arg("y"),
            py::pos_only(),
            py::arg("sample_weight") = std::nullopt)
        .def("get_params", [](const svc_type &self) {
            const plssvm::parameter params = self.svm_->get_params();

            py::dict py_params;
            py_params["C"] = params.cost.value();
            py_params["kernel"] = fmt::format("{}", params.kernel_type);
            py_params["degree"] = params.degree.value();
            py_params["gamma"] = params.gamma.value();  // TODO: scale, auto, value
            py_params["coef0"] = params.coef0.value();
            //            py_params["shrinking"];
            //            py_params["probability"];
            py_params["tol"] = self.epsilon.value_or(typename svc_type::real_type{ 1e-3 });
            //            py_params["cache_size"];
            //            py_params["class_weight"];
            py_params["verbose"] = plssvm::verbose;
            py_params["max_iter"] = self.max_iter.value_or(-1);
            //            py_params["decision_function_shape"];
            //            py_params["break_ties"];
            //            py_params["random_state"];

            return py_params;
        })
        .def("predict", [](svc_type &self, py::array_t<typename svc_type::real_type> data) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
            } else {
                const typename svc_type::data_set_type data_to_predict{ pyarray_to_vector_of_vector(data) };
                return self.svm_->predict(*self.model_, data_to_predict);
            }
        })
        .def("predict_log_proba", [](const svc_type &, py::array_t<typename svc_type::real_type>) {
            throw py::attribute_error{ "'SVC' object has no function 'predict_log_proba' (not implemented)" };
        })
        .def("predict_proba", [](const svc_type &, py::array_t<typename svc_type::real_type>) {
            throw py::attribute_error{ "'SVC' object has no function 'predict_proba' (not implemented)" };
        })
        .def(
            "score", [](svc_type &self, py::array_t<typename svc_type::real_type> data, py::array_t<typename svc_type::label_type> label, std::optional<std::vector<typename svc_type::real_type>> sample_weight) {
                if (sample_weight.has_value()) {
                    throw py::attribute_error{ "The 'sample_weight' parameter for a call to 'fit' is not implemented yet!" };
                }

                if (self.model_ == nullptr) {
                    throw py::attribute_error{ "This SVC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." };
                } else {
                    const typename svc_type::data_set_type data_to_score{ pyarray_to_vector_of_vector(data), pyarray_to_vector(label) };
                    return self.svm_->score(*self.model_, data_to_score);
                }
            },
            py::arg("X"),
            py::arg("y"),
            py::pos_only(),
            py::arg("sample_weight") = std::nullopt)
        .def("set_params", [](svc_type &self, py::kwargs args) {
            // TODO: remove copy-paste
            // check named arguments
            check_kwargs_for_correctness(args, { "C", "kernel", "degree", "gamma", "coef0", "shrinking", "probability", "tol", "cache_size", "class_weight", "verbose", "max_iter", "decision_function_shape", "break_ties", "random_state" });
            if (args.contains("C")) {
                self.svm_->set_params(plssvm::cost = args["C"].cast<typename svc_type::real_type>());
            }
            if (args.contains("kernel")) {
                std::stringstream ss{ args["kernel"].cast<std::string>() };
                plssvm::kernel_function_type kernel;
                ss >> kernel;
                self.svm_->set_params(plssvm::kernel_type = kernel);
            }
            if (args.contains("degree")) {
                self.svm_->set_params(plssvm::degree = args["degree"].cast<int>());
            }
            if (args.contains("gamma")) {
                self.svm_->set_params(plssvm::gamma = args["gamma"].cast<typename svc_type::real_type>());  // TODO: scale, auto, value
            }
            if (args.contains("coef0")) {
                self.svm_->set_params(plssvm::coef0 = args["coef0"].cast<typename svc_type::real_type>());
            }
            if (args.contains("shrinking")) {
                throw py::attribute_error{ "The 'shrinking' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("probability")) {
                throw py::attribute_error{ "The 'probability' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("tol")) {
                self.epsilon.value() = args["tol"].cast<typename svc_type::real_type>();
            }
            if (args.contains("cache_size")) {
                throw py::attribute_error{ "The 'cache_size' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("class_weight")) {
                throw py::attribute_error{ "The 'class_weight' parameter for a call to the 'SVC' constructor is not implemented yet!" };
            }
            if (args.contains("verbose")) {
                plssvm::verbose = args["verbose"].cast<bool>();
            }
            if (args.contains("max_iter")) {
                self.max_iter.value() = args["max_iter"].cast<long long>();
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
        })

        .def_property_readonly("class_weight_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'class_weight_' (not implemented)" };
        })
        .def_property_readonly("classes_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'fit_status_'" };
            } else {
                return self.data_->different_labels().value();
            }
        })
        .def_property_readonly("coef_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'coef_' (not implemented)" };
        })
        .def_property_readonly("coef_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'dual_coef_' (not implemented)" };
        })
        .def_property_readonly("fit_status_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'fit_status_'" };
            } else {
                return 0;
            }
        })
        .def_property_readonly("intercept_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'intercept_' (not implemented)" };
        })
        .def_property_readonly("n_features_in_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'n_features_in_'" };
            } else {
                return self.data_->num_features();
            }
        })
        .def_property_readonly("n_features_in_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'n_features_in_'" };
            } else {
                return self.data_->num_features();
            }
        })
        .def_property_readonly("feature_names_in_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'feature_names_in_' (not implemented)" };
        })
        .def_property_readonly("n_iter_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'n_iter_' (not implemented)" };
        })
        .def_property_readonly("support_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_'" };
            } else {
                // all data points are support vectors
                std::vector<std::size_t> indices(self.model_->num_support_vectors());  // TODO: type
                std::iota(indices.begin(), indices.end(), std::size_t{ 0 });
                return indices;
            }
        })
        .def_property_readonly("support_vectors_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'support_vectors_'" };
            } else {
                // all data points are support vectors
                return self.model_->support_vectors();
            }
        })
        .def_property_readonly("n_support_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'n_support_'" };
            } else {
                // all data points are support vectors
                std::map<typename svc_type::label_type, int> counts{};
                const std::vector<typename svc_type::label_type> different_labels = self.data_->different_labels().value();
                for (const typename svc_type::label_type &label : different_labels) {
                    counts[label] = 0;
                }
                const std::vector<typename svc_type::label_type> &labels = self.data_->labels()->get();
                const std::vector<typename svc_type::real_type> &weights = self.model_->weights();
                for (std::size_t i = 0; i < self.model_->num_support_vectors(); ++i) {
                    if (weights[i] != typename svc_type::real_type{ 0.0 }) {
                        ++counts[labels[i]];
                    }
                }

                std::vector<int> n_support(different_labels.size());
                for (std::size_t i = 0; i < n_support.size(); ++i) {
                    n_support[i] = counts[different_labels[i]];
                }
                return n_support;
            }
        })
        .def_property_readonly("probA_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'probA_' (not implemented)" };
        })
        .def_property_readonly("probB_", [](const svc_type &) {
            throw py::attribute_error{ "'SVC' object has no attribute 'probB_' (not implemented)" };
        })
        .def_property_readonly("shape_fit_", [](const svc_type &self) {
            if (self.model_ == nullptr) {
                throw py::attribute_error{ "'SVC' object has no attribute 'shape_fit_'" };
            } else {
                return std::tuple<int, int>{ static_cast<int>(self.data_->num_data_points()), static_cast<int>(self.data_->num_features()) };
            }
        });
}

// TODO: std::string as label type, reduce code duplication
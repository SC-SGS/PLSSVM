/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set.hpp"
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::real_type_label_type_combination_list

#include "utility.hpp"  // check_kwargs_for_correctness, assemble_unique_class_name,
                        // pyarray_to_vector, pyarray_to_string_vector, pylist_to_string_vector, pyarray_to_matrix

#include "fmt/core.h"           // fmt::format
#include "fmt/format.h"         // fmt::join
#include "pybind11/numpy.h"     // py::array_t
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::return_value_policy, py::arg, py::kwargs, py::value_error, py::pos_only, py::list
#include "pybind11/stl.h"       // support for STL types

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <tuple>        // std::tuple_element_t, std::tuple_size_v
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move, std::integer_sequence, std::make_integer_sequence
#include <vector>       // std::vector

namespace py = pybind11;

template <typename real_type, typename label_type>
void instantiate_data_set_bindings(py::module_ &m, plssvm::detail::real_type_label_type_combination<real_type, label_type>) {
    using data_set_type = plssvm::data_set<real_type, label_type>;
    using size_type = typename data_set_type::size_type;

    // create the Python type names based on the provided real_type and label_type
    const std::string class_name_scaling_factors = assemble_unique_class_name<real_type, label_type>("DataSetScalingFactors");
    const std::string class_name_scaling = assemble_unique_class_name<real_type, label_type>("DataSetScaling");
    const std::string class_name = assemble_unique_class_name<real_type, label_type>("DataSet");

    // bind the plssvm::data_set::scaling internal "factors" struct
    py::class_<typename data_set_type::scaling::factors>(m, class_name_scaling_factors.c_str())
        .def(py::init<size_type, real_type, real_type>(), "create a new scaling factor", py::arg("feature"), py::arg("lower"), py::arg("upper"))
        .def_readonly("feature", &data_set_type::scaling::factors::feature, "the feature index for which the factors are valid")
        .def_readonly("lower", &data_set_type::scaling::factors::lower, "the lower scaling factor")
        .def_readonly("upper", &data_set_type::scaling::factors::upper, "the upper scaling factor")
        .def("__repr__", [class_name_scaling_factors](const typename data_set_type::scaling::factors &scaling_factors) {
            return fmt::format("<plssvm.{} with {{ feature: {}, lower: {}, upper: {} }}>",
                               class_name_scaling_factors,
                               scaling_factors.feature,
                               scaling_factors.lower,
                               scaling_factors.upper);
        });

    // bind the plssvm::data_set internal "scaling" struct
    py::class_<typename data_set_type::scaling>(m, class_name_scaling.c_str())
        .def(py::init<real_type, real_type>(), "create new scaling factors for the range [lower, upper]", py::arg("lower"), py::arg("upper"))
        .def(py::init<const std::string &>(), "read the scaling factors from the file")
        .def("save", &data_set_type::scaling::save, "save the scaling factors to a file")
        .def_readonly("scaling_interval", &data_set_type::scaling::scaling_interval, "the interval to which the data points are scaled")
        .def_property_readonly(
            "scaling_factors", [](const typename data_set_type::scaling &scaling) {
                return vector_to_pyarray(scaling.scaling_factors);
            },
            "the scaling factors for each feature")
        .def("__repr__", [class_name_scaling](const typename data_set_type::scaling &scaling) {
            return fmt::format("<plssvm.{} with {{ lower: {}, upper: {}, #factors: {} }}>",
                               class_name_scaling,
                               scaling.scaling_interval.first,
                               scaling.scaling_interval.second,
                               scaling.scaling_factors.size());
        });

    // bind the data set class
    py::class_<data_set_type> py_data_set(m, class_name.c_str());
    // bind constructor taking a data set file
    py_data_set.def(py::init([](const std::string &file_name, py::kwargs args) {
                        // check for valid keys
                        check_kwargs_for_correctness(args, { "file_format", "scaling" });

                        // call the constructor corresponding to the provided named arguments
                        if (args.contains("file_format") && args.contains("scaling")) {
                            return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>(), args["scaling"].cast<typename data_set_type::scaling>() };
                        } else if (args.contains("file_format")) {
                            return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>() };
                        } else if (args.contains("scaling")) {
                            return data_set_type{ file_name, args["scaling"].cast<typename data_set_type::scaling>() };
                        } else {
                            return data_set_type{ file_name };
                        }
                    }),
                    "create a new data set from the provided file and additional optional parameters");
    // bind constructor taking only data points without labels
    py_data_set.def(py::init([](py::array_t<real_type> data, py::kwargs args) {
                        // check named arguments
                        check_kwargs_for_correctness(args, { "scaling" });

                        if (args.contains("scaling")) {
                            return data_set_type{ pyarray_to_matrix(data), args["scaling"].cast<typename data_set_type::scaling>() };
                        } else {
                            return data_set_type{ pyarray_to_matrix(data) };
                        }
                    }),
                    "create a new data set without labels given additional optional parameters");

    if constexpr (!std::is_same_v<label_type, std::string>) {
        py_data_set.def(py::init([](py::array_t<real_type> data, py::array_t<label_type> labels, py::kwargs args) {
                            // check named arguments
                            check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_vector(labels), args["scaling"].cast<typename data_set_type::scaling>() };
                            } else {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_vector(labels) };
                            }
                        }),
                        "create a new data set with labels from a numpy array given additional optional parameters");
    } else {
        // if the requested label_type is std::string, accept numpy arrays with real_type and convert them to a std::string internally
        py_data_set.def(py::init([](py::array_t<real_type> data, py::array_t<real_type> labels, py::kwargs args) {
                            // check named arguments
                            check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_string_vector(labels), args["scaling"].cast<typename data_set_type::scaling>() };
                            } else {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_string_vector(labels) };
                            }
                        }),
                        "create a new data set with labels from a numpy array given additional optional parameters");
        // if the requested label_type is std::string, accept a python list (which can contain py::str) and convert them to a std::string internally
        py_data_set.def(py::init([](py::array_t<real_type> data, py::list labels, py::kwargs args) {
                            // check named arguments
                            check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ pyarray_to_matrix(data), pylist_to_string_vector(labels), args["scaling"].cast<typename data_set_type::scaling>() };
                            } else {
                                return data_set_type{ pyarray_to_matrix(data), pylist_to_string_vector(labels) };
                            }
                        }),
                        "create a new data set with labels from a Python list given additional optional parameters");
    }

    py_data_set.def("save", &data_set_type::save, "save the data set to a file")
        .def("num_data_points", &data_set_type::num_data_points, "the number of data points in the data set")
        .def("num_features", &data_set_type::num_features, "the number of features per data point")
        .def("data", &data_set_type::data, py::return_value_policy::reference_internal, "the data saved as 2D vector")
        .def(
            "data", [](const data_set_type &data) {
                return matrix_to_pyarray(data.data());
            },
            "the data saved as 2D vector")
        .def("has_labels", &data_set_type::has_labels, "check whether the data set has labels")
        .def(
            "labels", [](const data_set_type &data) {
                if (!data.has_labels()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'labels'. Maybe this DataSet was created without labels?" };
                } else {
                    if constexpr (std::is_same_v<label_type, std::string>) {
                        return data.labels()->get();
                    } else {
                        return vector_to_pyarray(data.labels()->get());
                    }
                }
            },
            "the labels")
        .def("num_different_labels", &data_set_type::num_different_labels, "the number of different labels")
        .def(
            "different_labels", [](const data_set_type &data) {
                if (!data.has_labels()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'different_labels'. Maybe this DataSet was created without labels?" };
                } else {
                    if constexpr (std::is_same_v<label_type, std::string>) {
                        return data.different_labels().value();
                    } else {
                        return vector_to_pyarray(data.different_labels().value());
                    }
                }
            },
            "the different labels")
        .def("is_scaled", &data_set_type::is_scaled, "check whether the original data has been scaled to [lower, upper] bounds")
        .def(
            "scaling_factors", [](const data_set_type &data) {
                if (!data.is_scaled()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'scaling_factors'. Maybe this DataSet has not been scaled?" };
                } else {
                    return data.scaling_factors().value();
                }
            },
            py::return_value_policy::reference_internal,
            "the factors used to scale this data set")
        .def("__repr__", [class_name](const data_set_type &data) {
            std::string optional_repr{};
            if (data.has_labels()) {
                optional_repr += fmt::format(", labels: [{}]", fmt::join(data.different_labels().value(), ", "));
            }
            if (data.is_scaled()) {
                optional_repr += fmt::format(", scaling: [{}, {}]",
                                             data.scaling_factors()->get().scaling_interval.first,
                                             data.scaling_factors()->get().scaling_interval.second);
            }
            return fmt::format("<plssvm.{} with {{ #points: {}, #features: {}{} }}>",
                               class_name,
                               data.num_data_points(),
                               data.num_features(),
                               optional_repr);
        });
}

template <typename T, std::size_t... Idx>
void instantiate_data_set_bindings(py::module_ &m, std::integer_sequence<std::size_t, Idx...>) {
    (instantiate_data_set_bindings(m, std::tuple_element_t<Idx, T>{}), ...);
}

template <typename T>
void instantiate_data_set_bindings(py::module_ &m) {
    instantiate_data_set_bindings<T>(m, std::make_integer_sequence<std::size_t, std::tuple_size_v<T>>{});
}

void init_data_set(py::module_ &m) {
    // bind all data_set classes
    instantiate_data_set_bindings<plssvm::detail::real_type_label_type_combination_list>(m);

    // create aliases
    m.attr("DataSetScalingFactors") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_REAL_TYPE, PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("DataSetScalingFactors").c_str());
    m.attr("DataSetScaling") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_REAL_TYPE, PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("DataSetScaling").c_str());
    m.attr("DataSet") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_REAL_TYPE, PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("DataSet").c_str());
}
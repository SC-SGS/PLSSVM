/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"         // plssvm::real_type
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::label_type_list

#include "utility.hpp"  // check_kwargs_for_correctness, assemble_unique_class_name,
                        // pyarray_to_vector, pyarray_to_string_vector, pylist_to_string_vector, pyarray_to_matrix

#include "fmt/core.h"           // fmt::format
#include "fmt/format.h"         // fmt::join
#include "pybind11/numpy.h"     // py::array_t
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::return_value_policy, py::arg, py::kwargs, py::value_error, py::pos_only, py::list
#include "pybind11/stl.h"       // support for STL types

#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <tuple>        // std::tuple_element_t, std::tuple_size_v
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move, std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

template <typename data_set_type>
typename data_set_type::scaling create_scaling_object(const py::kwargs &args) {
    if (args.contains("scaling")) {
        typename data_set_type::scaling scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

        // try to directly convert it to a plssvm::data_set_type::scaling object
        try {
            scaling = args["scaling"].cast<typename data_set_type::scaling>();
        } catch (const py::cast_error &) {
            // can't cast to plssvm::data_set_type::scaling
            // -> try a std::array<real_type, 2> instead!
            try {
                const auto interval = args["scaling"].cast<std::array<plssvm::real_type, 2>>();
                scaling = typename data_set_type::scaling{ interval[0], interval[1] };
            } catch (...) {
                // rethrow exception if this also did not succeed
                throw;
            }
        }
        return scaling;
    } else {
        throw py::attribute_error{ "Can't extract scaling information, no scaling keyword argument given!" };
    }
}

template <typename label_type>
void instantiate_data_set_bindings(py::module_ &m, label_type) {
    using data_set_type = plssvm::data_set<label_type>;
    using size_type = typename data_set_type::size_type;

    // create the Python type names based on the provided real_type and label_type
    const std::string class_name_scaling_factors = assemble_unique_class_name<label_type>("DataSetScalingFactors");
    const std::string class_name_scaling = assemble_unique_class_name<label_type>("DataSetScaling");
    const std::string class_name = assemble_unique_class_name<label_type>("DataSet");

    PYBIND11_NUMPY_DTYPE(typename data_set_type::scaling::factors, feature, lower, upper);
    // bind the plssvm::data_set::scaling internal "factors" struct
    py::class_<typename data_set_type::scaling::factors>(m, class_name_scaling_factors.c_str())
        .def(py::init<size_type, plssvm::real_type, plssvm::real_type>(), "create a new scaling factor", py::arg("feature"), py::arg("lower"), py::arg("upper"))
        .def_readonly("feature", &data_set_type::scaling::factors::feature, "the feature index for which the factors are valid")
        .def_readonly("lower", &data_set_type::scaling::factors::lower, "the lower scaling factor")
        .def_readonly("upper", &data_set_type::scaling::factors::upper, "the upper scaling factor")
        .def("__repr__", [class_name_scaling_factors](const typename data_set_type::scaling::factors &self) {
            return fmt::format("<plssvm.{} with {{ feature: {}, lower: {}, upper: {} }}>",
                               class_name_scaling_factors,
                               self.feature,
                               self.lower,
                               self.upper);
        });

    // bind the plssvm::data_set internal "scaling" struct
    py::class_<typename data_set_type::scaling>(m, class_name_scaling.c_str())
        .def(py::init<plssvm::real_type, plssvm::real_type>(), "create new scaling factors for the range [lower, upper]", py::arg("lower"), py::arg("upper"))
        .def(py::init([](const std::array<plssvm::real_type, 2> interval) {
                 return typename data_set_type::scaling{ interval[0], interval[1] };
             }),
             "create new scaling factors for the range [lower, upper]")
        .def(py::init<const std::string &>(), "read the scaling factors from the file")
        .def("save", &data_set_type::scaling::save, "save the scaling factors to a file")
        .def_readonly("scaling_interval", &data_set_type::scaling::scaling_interval, "the interval to which the data points are scaled")
        .def_property_readonly(
            "scaling_factors", [](const typename data_set_type::scaling &scaling) {
                return vector_to_pyarray(scaling.scaling_factors);
            },
            "the scaling factors for each feature")
        .def("__repr__", [class_name_scaling](const typename data_set_type::scaling &self) {
            return fmt::format("<plssvm.{} with {{ lower: {}, upper: {}, #factors: {} }}>",
                               class_name_scaling,
                               self.scaling_interval.first,
                               self.scaling_interval.second,
                               self.scaling_factors.size());
        });

    // bind the data set class
    py::class_<data_set_type> py_data_set(m, class_name.c_str());
    // bind constructor taking a data set file
    py_data_set.def(py::init([](const std::string &file_name, py::kwargs args) {
                        // check for valid keys
                        check_kwargs_for_correctness(args, { "file_format", "scaling" });

                        // call the constructor corresponding to the provided keyword arguments
                        if (args.contains("file_format") && args.contains("scaling")) {
                            return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>(), create_scaling_object<data_set_type>(args) };
                        } else if (args.contains("file_format")) {
                            return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>() };
                        } else if (args.contains("scaling")) {
                            return data_set_type{ file_name, create_scaling_object<data_set_type>(args) };
                        } else {
                            return data_set_type{ file_name };
                        }
                    }),
                    "create a new data set from the provided file and additional optional parameters");
    // bind constructor taking only data points without labels
    py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, py::kwargs args) {
                        // check keyword arguments
                        check_kwargs_for_correctness(args, { "scaling" });

                        if (args.contains("scaling")) {
                            return data_set_type{ pyarray_to_matrix(data), create_scaling_object<data_set_type>(args) };
                        } else {
                            return data_set_type{ pyarray_to_matrix(data) };
                        }
                    }),
                    "create a new data set without labels given additional optional parameters");

    if constexpr (!std::is_same_v<label_type, std::string>) {
        py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<label_type, py::array::c_style | py::array::forcecast> labels, py::kwargs args) {
                            // check keyword arguments
                            check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_vector(labels), create_scaling_object<data_set_type>(args) };
                            } else {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_vector(labels) };
                            }
                        }),
                        "create a new data set with labels from a numpy array given additional optional parameters");
    } else {
        // if the requested label_type is std::string, accept numpy arrays with real_type and convert them to a std::string internally
        py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> labels, py::kwargs args) {
                            // check keyword arguments
                            check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_string_vector(labels), create_scaling_object<data_set_type>(args) };
                            } else {
                                return data_set_type{ pyarray_to_matrix(data), pyarray_to_string_vector(labels) };
                            }
                        }),
                        "create a new data set with labels from a numpy array given additional optional parameters");
        // if the requested label_type is std::string, accept a python list (which can contain py::str) and convert them to a std::string internally
        py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, const py::list &labels, py::kwargs args) {
                            // check keyword arguments
                            check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ pyarray_to_matrix(data), pylist_to_string_vector(labels), create_scaling_object<data_set_type>(args) };
                            } else {
                                return data_set_type{ pyarray_to_matrix(data), pylist_to_string_vector(labels) };
                            }
                        }),
                        "create a new data set with labels from a Python list given additional optional parameters");
    }

    py_data_set.def("save", py::overload_cast<const std::string &, plssvm::file_format_type>(&data_set_type::save, py::const_), "save the data set to a file using the provided file format type")
        .def("save", py::overload_cast<const std::string &>(&data_set_type::save, py::const_), "save the data set to a file automatically deriving the file format type from the file extension")
        .def("num_data_points", &data_set_type::num_data_points, "the number of data points in the data set")
        .def("num_features", &data_set_type::num_features, "the number of features per data point")
        .def(
            "data", [](const data_set_type &data) {
                return matrix_to_pyarray(data.data());
            },
            "the data saved as 2D vector")
        .def("has_labels", &data_set_type::has_labels, "check whether the data set has labels")
        .def(
            "labels", [](const data_set_type &self) {
                if (!self.has_labels()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'labels'. Maybe this DataSet was created without labels?" };
                } else {
                    if constexpr (std::is_same_v<label_type, std::string>) {
                        return self.labels()->get();
                    } else {
                        return vector_to_pyarray(self.labels()->get());
                    }
                }
            },
            "the labels")
        .def("num_classes", &data_set_type::num_classes, "the number of classes")
        .def(
            "classes", [](const data_set_type &self) {
                if (!self.has_labels()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'classes'. Maybe this DataSet was created without labels?" };
                } else {
                    if constexpr (std::is_same_v<label_type, std::string>) {
                        return self.classes().value();
                    } else {
                        return vector_to_pyarray(self.classes().value());
                    }
                }
            },
            "the classes")
        .def("is_scaled", &data_set_type::is_scaled, "check whether the original data has been scaled to [lower, upper] bounds")
        .def(
            "scaling_factors", [](const data_set_type &self) {
                if (!self.is_scaled()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'scaling_factors'. Maybe this DataSet has not been scaled?" };
                } else {
                    return self.scaling_factors().value();
                }
            },
            py::return_value_policy::reference_internal,
            "the factors used to scale this data set")
        .def("__repr__", [class_name](const data_set_type &self) {
            std::string optional_repr{};
            if (self.has_labels()) {
                optional_repr += fmt::format(", classes: [{}]", fmt::join(self.classes().value(), ", "));
            }
            if (self.is_scaled()) {
                optional_repr += fmt::format(", scaling: [{}, {}]",
                                             self.scaling_factors()->get().scaling_interval.first,
                                             self.scaling_factors()->get().scaling_interval.second);
            }
            return fmt::format("<plssvm.{} with {{ #points: {}, #features: {}{} }}>",
                               class_name,
                               self.num_data_points(),
                               self.num_features(),
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
    instantiate_data_set_bindings<plssvm::detail::supported_label_types>(m);

    // create aliases
    m.attr("DataSetScalingFactors") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("DataSetScalingFactors").c_str());
    m.attr("DataSetScaling") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("DataSetScaling").c_str());
    m.attr("DataSet") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("DataSet").c_str());
}
/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set

#include "plssvm/constants.hpp"          // plssvm::real_type
#include "plssvm/data_set/data_set.hpp"  // plssvm::data_set
#include "plssvm/detail/type_list.hpp"   // plssvm::detail::supported_label_types_classification
#include "plssvm/file_format_types.hpp"  // plssvm::file_format_type

#include "bindings/Python/data_set/utility.hpp"  // plssvm::bindings::python::util::create_scaling_object
#include "bindings/Python/utility.hpp"           // plssvm::bindings::python::util::{check_kwargs_for_correctness, assemble_unique_class_name, pyarray_t_to_vector, pyarray_t_to_string_vector, pyarray_to_matrix, instantiate_module_bindings}

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "pybind11/numpy.h"     // py::array_t
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::kwargs, py::list
#include "pybind11/stl.h"       // support for STL types

#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace py = pybind11;

/**
 * @brief Functor to instantiate all classification data set bindings.
 * @tparam label_type the label type for the classification data set
 */
template <typename label_type>
struct classification_data_set_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] m the base Python module
     * @param[in] pure_virtual the pure-virtual Python module
     */
    void operator()(py::module_ &m, py::module_ &pure_virtual, label_type) {
        using data_set_type = plssvm::classification_data_set<label_type>;

        // create the Python type names based on the provided label_type
        const std::string class_name = plssvm::bindings::python::util::assemble_unique_class_name<label_type>("ClassificationDataSet");
        m.attr(plssvm::bindings::python::util::assemble_unique_class_name<label_type>("ClassificationDataSetScalingFactors").c_str()) =
            pure_virtual.attr(plssvm::bindings::python::util::assemble_unique_class_name<label_type>("__pure_virtual_base_DataSetScalingFactors").c_str());
        m.attr(plssvm::bindings::python::util::assemble_unique_class_name<label_type>("ClassificationDataSetScaling").c_str()) =
            pure_virtual.attr(plssvm::bindings::python::util::assemble_unique_class_name<label_type>("__pure_virtual_base_DataSetScaling").c_str());

        // bind the data set class
        py::class_<data_set_type, plssvm::data_set<label_type>> py_data_set(m, class_name.c_str());
        // bind constructor taking a data set file
        py_data_set.def(py::init([](const std::string &file_name, py::kwargs args) {
                            // check for valid keys
                            plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "file_format", "scaling" });

                            // call the constructor corresponding to the provided keyword arguments
                            if (args.contains("file_format") && args.contains("scaling")) {
                                return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>(), plssvm::bindings::python::util::create_scaling_object<data_set_type>(args) };
                            } else if (args.contains("file_format")) {
                                return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>() };
                            } else if (args.contains("scaling")) {
                                return data_set_type{ file_name, plssvm::bindings::python::util::create_scaling_object<data_set_type>(args) };
                            } else {
                                return data_set_type{ file_name };
                            }
                        }),
                        "create a new data set from the provided file and additional optional parameters");
        // bind constructor taking only data points without labels
        py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, py::kwargs args) {
                            // check keyword arguments
                            plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "scaling" });

                            if (args.contains("scaling")) {
                                return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::create_scaling_object<data_set_type>(args) };
                            } else {
                                return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data) };
                            }
                        }),
                        "create a new data set without labels given additional optional parameters");

        if constexpr (!std::is_same_v<label_type, std::string>) {
            py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<label_type, py::array::c_style | py::array::forcecast> labels, py::kwargs args) {
                                // check keyword arguments
                                plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "scaling" });

                                if (args.contains("scaling")) {
                                    return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pyarray_t_to_vector(labels), plssvm::bindings::python::util::create_scaling_object<data_set_type>(args) };
                                } else {
                                    return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pyarray_t_to_vector(labels) };
                                }
                            }),
                            "create a new data set with labels from a numpy array given additional optional parameters");
        } else {
            // if the requested label_type is std::string, accept numpy arrays with real_type and convert them to a std::string internally
            py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> labels, py::kwargs args) {
                                // check keyword arguments
                                plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "scaling" });

                                if (args.contains("scaling")) {
                                    return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pyarray_t_to_string_vector(labels), plssvm::bindings::python::util::create_scaling_object<data_set_type>(args) };
                                } else {
                                    return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pyarray_t_to_string_vector(labels) };
                                }
                            }),
                            "create a new data set with labels from a numpy array given additional optional parameters");
            // if the requested label_type is std::string, accept a python list (which can contain py::str) and convert them to a std::string internally
            py_data_set.def(py::init([](py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast> data, const py::list &labels, py::kwargs args) {
                                // check keyword arguments
                                plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "scaling" });

                                if (args.contains("scaling")) {
                                    return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pylist_to_vector<std::string>(labels), plssvm::bindings::python::util::create_scaling_object<data_set_type>(args) };
                                } else {
                                    return data_set_type{ plssvm::bindings::python::util::pyarray_to_matrix(data), plssvm::bindings::python::util::pylist_to_vector<std::string>(labels) };
                                }
                            }),
                            "create a new data set with labels from a Python list given additional optional parameters");
        }

        // bind classification data set specific member functions
        py_data_set.def("num_classes", &data_set_type::num_classes, "the number of classes")
            .def("classes", [](const data_set_type &self) {
                if (!self.has_labels()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'classes'. Maybe this DataSet was created without labels?" };
                } else {
                    if constexpr (std::is_same_v<label_type, std::string>) {
                        return self.classes().value();
                    } else {
                        return plssvm::bindings::python::util::vector_to_pyarray(self.classes().value());
                    }
                } }, "the classes")
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
                                   optional_repr); });
    }
};

void init_classification_data_set(py::module_ &m, py::module_ &pure_virtual) {
    // bind all classification data_set classes
    plssvm::bindings::python::util::instantiate_module_bindings<classification_data_set_bindings, plssvm::detail::supported_label_types_classification>(m, pure_virtual);

    // create classification data set aliases
    m.attr("ClassificationDataSetScalingFactors") = m.attr(plssvm::bindings::python::util::assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_SVC_LABEL_TYPE>("ClassificationDataSetScalingFactors").c_str());
    m.attr("ClassificationDataSetScaling") = m.attr(plssvm::bindings::python::util::assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_SVC_LABEL_TYPE>("ClassificationDataSetScaling").c_str());
    m.attr("ClassificationDataSet") = m.attr(plssvm::bindings::python::util::assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_SVC_LABEL_TYPE>("ClassificationDataSet").c_str());
}

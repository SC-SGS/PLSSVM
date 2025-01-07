/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set/data_set.hpp"  // plssvm::data_set

#include "plssvm/constants.hpp"          // plssvm::real_type
#include "plssvm/detail/type_list.hpp"   // plssvm::detail::supported_label_types
#include "plssvm/file_format_types.hpp"  // plssvm::file_format_type

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, assemble_unique_class_name, vector_to_pyarray, instantiate_bindings}

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "pybind11/numpy.h"     // py::array_t
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::return_value_policy, py::arg, py::kwargs, py::value_error, py::pos_only, py::list
#include "pybind11/stl.h"       // support for STL types

#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <tuple>        // std::tuple_element_t, std::tuple_size_v
#include <type_traits>  // std::is_same_v
#include <utility>      // std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

/**
 * @brief Functor to instantiate all base data set bindings.
 * @tparam label_type the label type for the base data set
 */
template <typename label_type>
struct data_set_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] m the Python module
     */
    void operator()(py::module_ &m, label_type) {
        using data_set_type = plssvm::data_set<label_type>;
        using size_type = typename data_set_type::size_type;

        // create the Python type names based on the provided real_type and label_type
        const std::string class_name_scaling_factors = plssvm::bindings::python::util::assemble_unique_class_name<label_type>("__pure_virtual_base_DataSetScalingFactors");
        const std::string class_name_scaling = plssvm::bindings::python::util::assemble_unique_class_name<label_type>("__pure_virtual_base_DataSetScaling");
        const std::string class_name = plssvm::bindings::python::util::assemble_unique_class_name<label_type>("__pure_virtual_base_DataSet");

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
                    return plssvm::bindings::python::util::vector_to_pyarray(scaling.scaling_factors);
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
        py_data_set.def("save", py::overload_cast<const std::string &, plssvm::file_format_type>(&data_set_type::save, py::const_), "save the data set to a file using the provided file format type")
            .def("save", py::overload_cast<const std::string &>(&data_set_type::save, py::const_), "save the data set to a file automatically deriving the file format type from the file extension")
            .def("num_data_points", &data_set_type::num_data_points, "the number of data points in the data set")
            .def("num_features", &data_set_type::num_features, "the number of features per data point")
            .def("data", [](const data_set_type &data) { return plssvm::bindings::python::util::matrix_to_pyarray(data.data()); }, "the data saved as 2D vector")
            .def("has_labels", &data_set_type::has_labels, "check whether the data set has labels")
            .def("labels", [](const data_set_type &self) {
                if (!self.has_labels()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'labels'. Maybe this DataSet was created without labels?" };
                } else {
                    if constexpr (std::is_same_v<label_type, std::string>) {
                        return self.labels()->get();
                    } else {
                        return plssvm::bindings::python::util::vector_to_pyarray(self.labels()->get());
                    }
                } }, "the labels")
            .def("is_scaled", &data_set_type::is_scaled, "check whether the original data has been scaled to [lower, upper] bounds")
            .def("scaling_factors", [](const data_set_type &self) {
                if (!self.is_scaled()) {
                    throw py::attribute_error{ "'DataSet' object has no function 'scaling_factors'. Maybe this DataSet has not been scaled?" };
                } else {
                    return self.scaling_factors().value();
                } }, py::return_value_policy::reference_internal, "the factors used to scale this data set");
    }
};

void init_data_set(py::module_ &pure_virtual) {
    // bind pure-virtual base data_set classes
    // NOTE: supported_label_types_classification also contains all types in supported_label_types_regression
    plssvm::bindings::python::util::instantiate_bindings<data_set_bindings, plssvm::detail::supported_label_types_classification>(pure_virtual);
}

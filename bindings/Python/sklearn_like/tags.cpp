/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "bindings/Python/sklearn_like/tags.hpp"  // Tags, TargetTags, TransformerTags, ClassifierTags, RegressorTags, InputTags

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "pybind11/cast.h"      // py::cast, py::repr
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // NOLINT: bind STL types

#include <string>  // std::string

namespace {

[[nodiscard]] std::string bool_as_python_string(const bool b) {
    return b ? "True" : "False";
}

}  // namespace

void init_sklearn_tags(py::module_ &m) {
    // TargetTags
    py::class_<TargetTags>(m, "TargetTags")
        .def(py::init<>())
        .def_readwrite("required", &TargetTags::required)
        .def_readwrite("one_d_labels", &TargetTags::one_d_labels)
        .def_readwrite("two_d_labels", &TargetTags::two_d_labels)
        .def_readwrite("positive_only", &TargetTags::positive_only)
        .def_readwrite("multi_output", &TargetTags::multi_output)
        .def_readwrite("single_output", &TargetTags::single_output)
        .def("__repr__", [](const TargetTags &t) {
            return fmt::format(
                "TargetTags(required={}, one_d_labels={}, two_d_labels={}, "
                "positive_only={}, multi_output={}, single_output={})",
                bool_as_python_string(t.required),
                bool_as_python_string(t.one_d_labels),
                bool_as_python_string(t.two_d_labels),
                bool_as_python_string(t.positive_only),
                bool_as_python_string(t.multi_output),
                bool_as_python_string(t.single_output));
        });

    // TransformerTags
    py::class_<TransformerTags>(m, "TransformerTags")
        .def(py::init<>())
        .def_readwrite("preserves_dtype", &TransformerTags::preserves_dtype)
        .def("__repr__", [](const TransformerTags &t) {
            return fmt::format(
                "TransformerTags(preserves_dtype=[{}])",
                fmt::join(t.preserves_dtype, ", "));
        });

    // ClassifierTags
    py::class_<ClassifierTags>(m, "ClassifierTags")
        .def(py::init<>())
        .def_readwrite("poor_score", &ClassifierTags::poor_score)
        .def_readwrite("multi_class", &ClassifierTags::multi_class)
        .def_readwrite("multi_label", &ClassifierTags::multi_label)
        .def("__repr__", [](const ClassifierTags &t) {
            return fmt::format(
                "ClassifierTags(poor_score={}, multi_class={}, multi_label={})",
                bool_as_python_string(t.poor_score),
                bool_as_python_string(t.multi_class),
                bool_as_python_string(t.multi_label));
        });

    // RegressorTags
    py::class_<RegressorTags>(m, "RegressorTags")
        .def(py::init<>())
        .def_readwrite("poor_score", &RegressorTags::poor_score)
        .def("__repr__", [](const RegressorTags &t) {
            return fmt::format(
                "RegressorTags(poor_score={})",
                bool_as_python_string(t.poor_score));
        });

    // InputTags
    py::class_<InputTags>(m, "InputTags")
        .def(py::init<>())
        .def_readwrite("one_d_array", &InputTags::one_d_array)
        .def_readwrite("two_d_array", &InputTags::two_d_array)
        .def_readwrite("three_d_array", &InputTags::three_d_array)
        .def_readwrite("sparse", &InputTags::sparse)
        .def_readwrite("categorical", &InputTags::categorical)
        .def_readwrite("string", &InputTags::string)
        .def_readwrite("dict", &InputTags::dict)
        .def_readwrite("positive_only", &InputTags::positive_only)
        .def_readwrite("allow_nan", &InputTags::allow_nan)
        .def_readwrite("pairwise", &InputTags::pairwise)
        .def("__repr__", [](const InputTags &t) {
            return fmt::format(
                "InputTags(one_d_array={}, two_d_array={}, three_d_array={}, "
                "sparse={}, categorical={}, string={}, dict={}, "
                "positive_only={}, allow_nan={}, pairwise={})",
                bool_as_python_string(t.one_d_array),
                bool_as_python_string(t.two_d_array),
                bool_as_python_string(t.three_d_array),
                bool_as_python_string(t.sparse),
                bool_as_python_string(t.categorical),
                bool_as_python_string(t.string),
                bool_as_python_string(t.dict),
                bool_as_python_string(t.positive_only),
                bool_as_python_string(t.allow_nan),
                bool_as_python_string(t.pairwise));
        });

    // Tags (root object)
    py::class_<Tags>(m, "Tags")
        .def(py::init<>())
        .def_readwrite("estimator_type", &Tags::estimator_type)
        .def_readwrite("target_tags", &Tags::target_tags)
        .def_readwrite("transformer_tags", &Tags::transformer_tags)
        .def_readwrite("classifier_tags", &Tags::classifier_tags)
        .def_readwrite("regressor_tags", &Tags::regressor_tags)
        .def_readwrite("array_api_support", &Tags::array_api_support)
        .def_readwrite("no_validation", &Tags::no_validation)
        .def_readwrite("non_deterministic", &Tags::non_deterministic)
        .def_readwrite("requires_fit", &Tags::requires_fit)
        .def_readwrite("_skip_test", &Tags::_skip_test)
        .def_readwrite("input_tags", &Tags::input_tags)
        .def("__repr__", [](const Tags &t) {
            const std::string estimator_type = t.estimator_type.has_value()
                                                   ? fmt::format("'{}'", t.estimator_type.value())
                                                   : "None";
            const auto &tag_as_string_or_none = [](const auto &opt_tag) -> std::string {
                if (!opt_tag.has_value()) {
                    return "None";
                }
                return py::repr(py::cast(opt_tag.value())).template cast<std::string>();
            };

            return fmt::format(
                "Tags(estimator_type={}, target_tags={}, transformer_tags={}, "
                "classifier_tags={}, regressor_tags={}, array_api_support={}, "
                "no_validation={}, non_deterministic={}, requires_fit={}, "
                "_skip_test={}, input_tags={})",
                estimator_type,
                py::repr(py::cast(t.target_tags)).cast<std::string>(),
                tag_as_string_or_none(t.transformer_tags),
                tag_as_string_or_none(t.classifier_tags),
                tag_as_string_or_none(t.regressor_tags),
                bool_as_python_string(t.array_api_support),
                bool_as_python_string(t.no_validation),
                bool_as_python_string(t.non_deterministic),
                bool_as_python_string(t.requires_fit),
                bool_as_python_string(t._skip_test),
                py::repr(py::cast(t.input_tags)).cast<std::string>());
        });
}

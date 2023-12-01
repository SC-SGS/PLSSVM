/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/performance_tracker.hpp"

#include "pybind11/pybind11.h"  // py::module_

namespace py = pybind11;

void init_performance_tracker([[maybe_unused]] py::module_ &m) {
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

    // use a detail.PerformanceTracker submodule for the performance tracking bindings
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality for the performance tracker");
    py::module_ performance_tracker_module = detail_module.def_submodule("PerformanceTracker");

    // bind the performance tracker functions
    performance_tracker_module
        .def("add_string_tracking_entry", [](const std::string &category, const std::string &name, const std::string &value) {
            plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ category, name, value });
        })
        .def("add_parameter_tracking_entry", [](const plssvm::parameter &params) {
            plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "parameter", "", params });
        })
        .def(
            "pause", []() { plssvm::detail::global_tracker->pause_tracking(); }, "pause performance tracking")
        .def(
            "resume", []() { plssvm::detail::global_tracker->resume_tracking(); }, "resume performance tracking")
        .def(
            "save", [](const std::string &filename) { plssvm::detail::global_tracker->save(filename); }, "save the performance tracking results to the specified yaml file")
        .def(
            "is_tracking", []() { return plssvm::detail::global_tracker->is_tracking(); }, "check whether performance tracking is currently enabled")
        .def(
            "clear_tracking_entries", []() { plssvm::detail::global_tracker->clear_tracking_entries(); }, "remove all currently tracked entries from the performance tracker");

#endif
}
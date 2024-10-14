/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::{global_tracker, tracking_entry}

#include "plssvm/detail/tracking/events.hpp"            // plssvm::detail::tracking::events
#include "plssvm/parameter.hpp"                         // plssvm::parameter

#include "pybind11/chrono.h"    // automatic bindings for std::chrono::milliseconds
#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // automatic bindings for std::optional and std::vector

#include <chrono>  // std::chrono::steady_clock::time_point
#include <string>  // std::string

namespace py = pybind11;

void init_performance_tracker(py::module_ &m) {
    // use a detail.tracking.PerformanceTracker submodule for the performance tracking bindings
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");
    py::module_ performance_tracker_module = tracking_module.def_submodule("PerformanceTracker");

    // bind the performance tracker functions
    performance_tracker_module
        // clang-format off
        .def("add_string_tracking_entry", [](const std::string &category, const std::string &name, const std::string &value) {
            plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ category, name, value });
        }, "add a new generic string tracking entry")
        .def("add_parameter_tracking_entry", [](const plssvm::parameter &params) {
            plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "parameter", "", params });
        }, "add a new parameter tracking entry")
        // clang-format on
        .def("add_event", [](const std::string &name) { plssvm::detail::tracking::global_performance_tracker().add_event(name); }, "add a new event")
        .def("pause", []() { plssvm::detail::tracking::global_performance_tracker().pause_tracking(); }, "pause performance tracking")
        .def("resume", []() { plssvm::detail::tracking::global_performance_tracker().resume_tracking(); }, "resume performance tracking")
        .def("save", [](const std::string &filename) { plssvm::detail::tracking::global_performance_tracker().save(filename); }, "save the performance tracking results to the specified yaml file")
        .def("set_reference_time", [](const std::chrono::steady_clock::time_point time) { plssvm::detail::tracking::global_performance_tracker().set_reference_time(time); }, "set a new reference time")
        .def("get_reference_time", []() { return plssvm::detail::tracking::global_performance_tracker().get_reference_time(); }, "get the current reference time")
        .def("is_tracking", []() { return plssvm::detail::tracking::global_performance_tracker().is_tracking(); }, "check whether performance tracking is currently enabled")
        .def("get_tracking_entries", []() { return plssvm::detail::tracking::global_performance_tracker().get_tracking_entries(); }, py::return_value_policy::reference, "retrieve all currently added tracking entries")
        .def("get_events", []() { return plssvm::detail::tracking::global_performance_tracker().get_events(); }, py::return_value_policy::reference, "retrieve all currently added events")
        .def("clear_tracking_entries", []() { plssvm::detail::tracking::global_performance_tracker().clear_tracking_entries(); }, "remove all currently tracked entries from the performance tracker");
}

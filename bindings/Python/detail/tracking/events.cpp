/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/events.hpp"  // plssvm::detail::tracking::events

#include "fmt/chrono.h"         // format std::chrono types
#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // bind STL types

namespace py = pybind11;

void init_events(py::module_ &m) {
    // use a detail.tracking.PerformanceTracker submodule for the performance tracking bindings
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");
    const py::module_ performance_tracker_module = tracking_module.def_submodule("PerformanceTracker");

    using event_type = plssvm::detail::tracking::events::event;

    // bind a single event
    py::class_<event_type>(performance_tracker_module, "Event")
        .def(py::init<decltype(event_type::time_point), decltype(event_type::name)>(), "construct a new event using a time point and a name")
        .def_readonly("time_point", &event_type::time_point, "read the time point associated to this event")
        .def_readonly("name", &event_type::name, "read the name associated to this event")
        .def("__repr__", [](const event_type &self) {
            return fmt::format("<plssvm.detail.tracking.Events.Event with {{ time_point: {}, name: {} }}>", self.time_point.time_since_epoch(), self.name);
        });

    // bind the events wrapper
    py::class_<plssvm::detail::tracking::events>(performance_tracker_module, "Events")
        .def(py::init<>(), "construct an empty events wrapper")
        .def("add_event", py::overload_cast<event_type>(&plssvm::detail::tracking::events::add_event), "add a new event")
        .def("add_event", py::overload_cast<decltype(event_type::time_point), decltype(event_type::name)>(&plssvm::detail::tracking::events::add_event), "add a new event using a time point and a name")
        .def("at", &plssvm::detail::tracking::events::operator[], "get the i-th event")
        .def("num_events", &plssvm::detail::tracking::events::num_events, "get the number of events")
        .def("empty", &plssvm::detail::tracking::events::empty, "check whether there are any events")
        .def("get_time_points", &plssvm::detail::tracking::events::get_time_points, "get all stored time points")
        .def("get_names", &plssvm::detail::tracking::events::get_names, "get all stored names")
        .def("__repr__", [](const plssvm::detail::tracking::events &self) {
            return fmt::format("<plssvm.detail.tracking.Events with\n{}\n>", self);
        });
}

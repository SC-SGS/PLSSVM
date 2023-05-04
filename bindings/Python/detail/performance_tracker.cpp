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

void init_performance_tracker(py::module_ &m) {
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

    // use a detail submodule for the performance tracking bindings
    const py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality for the performance tracker");

    // bind the performance tracker
    py::class_<plssvm::detail::performance_tracker>(detail_module, "PerformanceTracker")
        .def_static("add_string_tracking_entry", [](const std::string &category, const std::string & name, const std::string &value) {
            plssvm::detail::performance_tracker::add_tracking_entry(plssvm::detail::tracking_entry{ category, name, value });
        })
        .def_static("add_parameter_tracking_entry", [](const plssvm::parameter &params) {
            plssvm::detail::performance_tracker::add_tracking_entry(plssvm::detail::tracking_entry{ "parameter", "", params });
        })
        .def_static("pause", &plssvm::detail::performance_tracker::pause_tracking, "pause performance tracking")
        .def_static("resume", &plssvm::detail::performance_tracker::resume_tracking, "resume performance tracking")
        .def_static("save_to", py::overload_cast<const std::string &>(&plssvm::detail::performance_tracker::save), "save the performance tracking results to the specified yaml file");

#endif
}
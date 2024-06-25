/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception, plssvm::hardware_sampler_exception

#include "bindings/Python/utility.hpp"  // register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::class_

namespace py = pybind11;

void init_hardware_sampler(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    const py::module_ pure_virtual_model = m.def_submodule("__pure_virtual");

    // bind the pure virtual hardware sampler base class
    py::class_<plssvm::detail::tracking::hardware_sampler>(pure_virtual_model, "__pure_virtual_base_HardwareSampler")
        .def("start", &plssvm::detail::tracking::hardware_sampler::start_sampling, "start the current hardware sampling")
        .def("stop", &plssvm::detail::tracking::hardware_sampler::stop_sampling, "stop the current hardware sampling")
        .def("pause", &plssvm::detail::tracking::hardware_sampler::pause_sampling, "pause the current hardware sampling")
        .def("resume", &plssvm::detail::tracking::hardware_sampler::resume_sampling, "resume the current hardware sampling")
        .def("has_started", &plssvm::detail::tracking::hardware_sampler::has_sampling_started, "check whether hardware sampling has already been started")
        .def("is_sampling", &plssvm::detail::tracking::hardware_sampler::is_sampling, "check whether the hardware sampling is currently active")
        .def("has_stopped", &plssvm::detail::tracking::hardware_sampler::has_sampling_stopped, "check whether hardware sampling has already been stopped")
        .def("time_points", &plssvm::detail::tracking::hardware_sampler::time_points, "get the time points of the respective hardware samples")
        .def("sampling_interval", &plssvm::detail::tracking::hardware_sampler::sampling_interval, "get the sampling interval of this hardware sampler (in ms)");

    // register hardware sampler specific exception
    register_py_exception<plssvm::hardware_sampling_exception>(m, "HardwareSamplerError", base_exception);
}

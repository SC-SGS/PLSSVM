/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler

#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    #include "plssvm/detail/tracking/cpu/hardware_sampler.hpp"  // plssvm::detail::tracking::cpu_hardware_sampler
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
    #include "plssvm/detail/tracking/gpu_amd/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_amd_hardware_sampler
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
    #include "plssvm/detail/tracking/gpu_intel/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_intel_hardware_sampler
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
    #include "plssvm/detail/tracking/gpu_nvidia/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_nvidia_hardware_sampler
#endif
#include "plssvm/detail/tracking/hardware_sampler_factory.hpp"  // plssvm::detail::tracking::make_hardware_sampler
#include "plssvm/detail/utility.hpp"                            // plssvm::detail::unreachable
#include "plssvm/exceptions/exceptions.hpp"                     // plssvm::exception, plssvm::hardware_sampler_exception
#include "plssvm/target_platforms.hpp"                          // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::class_

#include <chrono>   // std::chrono::milliseconds
#include <cstddef>  // std::size_t

namespace py = pybind11;

void init_hardware_sampler(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    const py::module_ pure_virtual_model = m.def_submodule("__pure_virtual");

    // use a detail.tracking submodule for the hardware sampling
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    const py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");

    // bind the pure virtual hardware sampler base class
    py::class_<plssvm::detail::tracking::hardware_sampler> pyhardware_sampler(pure_virtual_model, "__pure_virtual_base_HardwareSampler");
    pyhardware_sampler.def("start", &plssvm::detail::tracking::hardware_sampler::start_sampling, "start the current hardware sampling")
        .def("stop", &plssvm::detail::tracking::hardware_sampler::stop_sampling, "stop the current hardware sampling")
        .def("pause", &plssvm::detail::tracking::hardware_sampler::pause_sampling, "pause the current hardware sampling")
        .def("resume", &plssvm::detail::tracking::hardware_sampler::resume_sampling, "resume the current hardware sampling")
        .def("has_started", &plssvm::detail::tracking::hardware_sampler::has_sampling_started, "check whether hardware sampling has already been started")
        .def("is_sampling", &plssvm::detail::tracking::hardware_sampler::is_sampling, "check whether the hardware sampling is currently active")
        .def("has_stopped", &plssvm::detail::tracking::hardware_sampler::has_sampling_stopped, "check whether hardware sampling has already been stopped")
        .def("time_points", &plssvm::detail::tracking::hardware_sampler::time_points, "get the time points of the respective hardware samples")
        .def("sampling_interval", &plssvm::detail::tracking::hardware_sampler::sampling_interval, "get the sampling interval of this hardware sampler (in ms)")
        .def("sampling_target", &plssvm::detail::tracking::hardware_sampler::sampling_target, "get the sampling target of this hardware sampler");

    // bind plssvm::make_csvm factory function to a "generic" Python csvm class
    py::class_<plssvm::detail::tracking::hardware_sampler>(tracking_module, "HardwareSampler", pyhardware_sampler, py::module_local())
        .def(py::init([](const plssvm::target_platform target, const std::size_t device_id, const std::chrono::milliseconds sampling_interval) {
                 return plssvm::detail::tracking::make_hardware_sampler(target, device_id, sampling_interval);
             }),
             "create a hardware sampler for the provided target using the given device ID and sampling interval")
        .def(py::init([](const plssvm::target_platform target, const std::size_t device_id) {
                 return plssvm::detail::tracking::make_hardware_sampler(target, device_id);
             }),
             "create a hardware sampler for the provided target using the given device ID")
        .def("__repr__", [](const plssvm::detail::tracking::hardware_sampler &self) {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
            if (dynamic_cast<const plssvm::detail::tracking::cpu_hardware_sampler *>(&self)) {
                return fmt::format("<plssvm.detail.tracking.CpuHardwareSampler with\n{}\n>", dynamic_cast<const plssvm::detail::tracking::cpu_hardware_sampler &>(self));
            }
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
            if (dynamic_cast<const plssvm::detail::tracking::gpu_nvidia_hardware_sampler *>(&self)) {
                return fmt::format("<plssvm.detail.tracking.GpuNvidiaHardwareSampler with\n{}\n>", dynamic_cast<const plssvm::detail::tracking::gpu_nvidia_hardware_sampler &>(self));
            }
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
            if (dynamic_cast<const plssvm::detail::tracking::gpu_amd_hardware_sampler *>(&self)) {
                return fmt::format("<plssvm.detail.tracking.GpuAmdHardwareSampler with\n{}\n>", dynamic_cast<const plssvm::detail::tracking::gpu_amd_hardware_sampler &>(self));
            }
#endif
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
            if (dynamic_cast<const plssvm::detail::tracking::gpu_intel_hardware_sampler *>(&self)) {
                return fmt::format("<plssvm.detail.tracking.GpuIntelHardwareSampler with\n{}\n>", dynamic_cast<const plssvm::detail::tracking::gpu_intel_hardware_sampler &>(self));
            }
#endif
            // unreachable!
            plssvm::detail::unreachable();
        });

    // register hardware sampler specific exception
    register_py_exception<plssvm::hardware_sampling_exception>(m, "HardwareSamplerError", base_exception);
}

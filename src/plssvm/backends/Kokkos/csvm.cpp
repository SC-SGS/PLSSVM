/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/csvm.hpp"

#include "plssvm/backends/Kokkos/detail/execution_space.hpp"  // plssvm::kokkos::detail::execution_space
#include "plssvm/backends/Kokkos/detail/utility.hpp"          // plssvm::kokkos::detail::get_runtime_version
#include "plssvm/backends/Kokkos/exceptions.hpp"              // plssvm::kokkos::backend_exception
#include "plssvm/detail/logging.hpp"                          // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"     // plssvm::detail::tracking::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                   // plssvm::exception
#include "plssvm/parameter.hpp"                               // plssvm::parameter
#include "plssvm/target_platforms.hpp"                        // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                        // plssvm::verbosity_level

#include "Kokkos_Core.hpp"  // TODO:

#include "fmt/format.h"  // fmt::format

#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <numeric>    // std::iota
#include <vector>     // std::vector

namespace plssvm::kokkos {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } { }

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check whether the requested target platform has been enabled
    switch (target) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
    }

    // TODO: document: we ALWAYS use the default execution space

    // set the execution space -> we always only use the Kokkos::DefaultExecutionSpace
    space_ = detail::determine_execution_space<Kokkos::DefaultExecutionSpace>();

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing Kokkos ({}) as backend with the Kokkos::DefaultExecutionSpace \"{}\".\n",
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "kokkos_version", detail::get_kokkos_version() },
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "kokkos_default_execution_space", space_ });

    // check whether the provided target platform is compatible with the Kokkos execution space
    if (target == target_platform::automatic) {
        // determine the default target based on the provided Kokkos execution space
        target_ = detail::determine_default_target_platform_from_execution_space(space_);
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    } else {
        // check whether the provided target platform is compatible with the execution space
        // throws a backend exception if the combination is invalid
        detail::check_execution_space_target_platform_combination(space_, target);
        target_ = target;
    }

    // get all available devices wrt the requested target platform
    devices_.resize(static_cast<std::vector<queue_type>::size_type>(Kokkos::num_devices()));
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no CUDA devices could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("Not devices found for the Kokkos execution space {} with the target platform {}!", space_, target_) };
    }

    // print found Kokkos devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} Kokkos device(s) for the target platform {}:\n",
                        plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ });

    std::vector<std::string> device_names;
    device_names.reserve(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::string device_name = detail::get_device_name(space_, device);
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}]\n",
                            device,
                            device_name);
        device_names.emplace_back(device_name);
    }
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_names }));
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the Kokkos execution spaces have finished before destruction
        detail::device_synchronize_all();
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    return {};
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    return {};
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    return {};
}

::plssvm::detail::dim_type csvm::get_max_grid_size([[maybe_unused]] const std::size_t device_id) const {
    return {};
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    return {};
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const ::plssvm::detail::execution_range &mirror_exec, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    return {};
}

auto csvm::run_predict_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    return {};
}

}  // namespace plssvm::kokkos

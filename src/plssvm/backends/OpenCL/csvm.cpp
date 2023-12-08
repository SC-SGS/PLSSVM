/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/csvm.hpp"

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/context.hpp"        // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"     // plssvm::opencl::detail::device_ptr
#include "plssvm/backends/OpenCL/detail/kernel.hpp"         // plssvm::opencl::detail::{compute_kernel_name, kernel}
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // plssvm::opencl::detail::{get_contexts, create_command_queues, run_kernel, kernel_type_to_function_name, device_synchronize}
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                             // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                        // plssvm::detail::log
#include "plssvm/detail/performance_tracker.hpp"            // plssvm::detail::tracking_entry
#include "plssvm/detail/utility.hpp"                        // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                      // plssvm::verbosity_level

#include "fmt/chrono.h"   // can directly print std::chrono literals
#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>  // std::all_of
#include <chrono>     // std::chrono
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <string>     // std::string
#include <tuple>      // std::tie
#include <utility>    // std::pair, std::make_pair, std::move
#include <vector>     // std::vector

namespace plssvm::opencl {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
}

csvm::~csvm() {
    try {
        // be sure that all operations on the OpenCL devices have finished before destruction
        for (const queue_type &queue : devices_) {
            detail::device_synchronize(queue);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
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

    // get kernel type from base class
    const kernel_function_type kernel = base_type::get_params().kernel_type;

    // get all available OpenCL contexts for the current target including devices with respect to the requested target platform
    std::tie(contexts_, target_) = detail::get_contexts(target);

    // currently, only a single context is allowed
    if (contexts_.size() > 1) {
        throw backend_exception{ fmt::format("Currently only a single OpenCL context is allowed, but {} were given!", contexts_.size()) };
    }

    // throw exception if no devices for the requested target could be found
    if (contexts_[0].devices.empty()) {
        throw backend_exception{ fmt::format("OpenCL backend selected but no devices for the target {} were found!", target) };
    }

    // print OpenCL info
    plssvm::detail::log(verbosity_level::full,
                        "\nUsing OpenCL as backend.\n");
    if (target == target_platform::automatic) {
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::opencl }));

    // create command_queues and JIT compile OpenCL kernels
    const auto jit_start_time = std::chrono::steady_clock::now();

    // get kernel names
    const std::vector<std::pair<detail::compute_kernel_name, std::string>> kernel_names = detail::kernel_type_to_function_names(kernel);
    // compile all kernels for float and double
    devices_ = detail::create_command_queues(contexts_, target_, kernel_names);

    // currently only single GPU execution is supported
    if (devices_.size() != 1) {
        plssvm::detail::log(verbosity_level::full | verbosity_level::warning,
                            "WARNING: found {} devices, but currently only single GPU execution is supported. Continuing only with device 0!\n",
                            devices_.size());
        devices_.resize(1);
    }

    const auto jit_end_time = std::chrono::steady_clock::now();
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\nOpenCL kernel JIT compilation done in {}.\n",
                        plssvm::detail::tracking_entry{ "backend", "jit_compilation_time", std::chrono::duration_cast<std::chrono::milliseconds>(jit_end_time - jit_start_time) });

    // print found OpenCL devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} OpenCL device(s) for the target platform {}:\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking_entry{ "backend", "target_platform", target_ });
    std::vector<std::string> device_names;
    device_names.reserve(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::string device_name = detail::get_device_name(devices_[device]);
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}]\n",
                            device,
                            device_name);
        device_names.emplace_back(device_name);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "device", device_names }));
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");

    // sanity checks for the number of the OpenCL kernels
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.size() == 6; }), "Every command queue must have exactly four associated kernels!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::assemble_kernel_matrix_explicit); }),
                  "The explicit kernel matrix assembly device kernel is missing!");
#if defined(PLSSVM_USE_GEMM)
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::gemm_kernel_explicit); }),
                  "The explicit BLAS GEMM device kernel is missing!");
#else
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::symm_kernel_explicit); }),
                  "The explicit BLAS SYMM device kernel is missing!");
#endif
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::w_kernel); }),
                  "The w_kernel device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_linear); }),
                  "The predict_kernel_linear device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_polynomial); }),
                  "The predict_kernel_polynomial device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_rbf); }),
                  "The predict_kernel_rbf device kernel is missing!");
}

void csvm::device_synchronize(const queue_type &queue) const {
    detail::device_synchronize(queue);
}

::plssvm::detail::memory_size csvm::get_device_memory() const {
    // get device
    cl_device_id device_id{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(devices_[0], CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");

    // get device global memory size
    cl_ulong total_device_memory{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &total_device_memory, nullptr), "error obtaining device's global memory size");
    return ::plssvm::detail::memory_size{ static_cast<unsigned long long>(total_device_memory) };
}

::plssvm::detail::memory_size csvm::get_max_mem_alloc_size() const {
    // get device
    cl_device_id device_id{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(devices_[0], CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");

    // get maximum allocation size
    cl_ulong max_alloc_size{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, nullptr), "error obtaining device's maximum allocation size");
    return ::plssvm::detail::memory_size{ static_cast<unsigned long long>(max_alloc_size) };
}

std::size_t csvm::get_max_work_group_size() const {
    // get device
    cl_device_id device_id{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(devices_[0], CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");
    // get maximum work group size
    cl_ulong max_work_group_size{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &max_work_group_size, nullptr), "error obtaining device's global memory size");
    return static_cast<std::size_t>(max_work_group_size);
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const cl_ulong num_rows_reduced = data_d.size(0) - 1;
    const cl_ulong num_features = data_d.size(1);

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const std::vector<std::size_t> block = { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const std::vector<std::size_t> grid = { static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                            static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };

#if defined(PLSSVM_USE_GEMM)
    device_ptr_type kernel_matrix_d{ (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE), devices_[0] };  // store full matrix
#else
    device_ptr_type kernel_matrix_d{ (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2, devices_[0] };  // only explicitly store the upper triangular matrix
#endif
    kernel_matrix_d.memset(0);
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), grid, block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor);
            break;
        case kernel_function_type::polynomial:
            detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), grid, block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), grid, block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value());
            break;
    }
    this->device_synchronize(devices_[0]);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t m, const std::size_t n, const std::size_t k, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const std::vector<std::size_t> block = { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const std::vector<std::size_t> grid = { static_cast<std::size_t>(std::ceil(static_cast<double>(n) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                            static_cast<std::size_t>(std::ceil(static_cast<double>(m) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };

    // cast to correct type
    const auto m_ull = static_cast<cl_ulong>(m);
    const auto n_ull = static_cast<cl_ulong>(n);
    const auto k_ull = static_cast<cl_ulong>(k);
#if defined(PLSSVM_USE_GEMM)
    detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::gemm_kernel_explicit), grid, block, m_ull, n_ull, k_ull, alpha, A_d.get(), B_d.get(), beta, C_d.get());
#else
    detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::symm_kernel_explicit), grid, block, m_ull, n_ull, k_ull, alpha, A_d.get(), B_d.get(), beta, C_d.get());
#endif
    this->device_synchronize(devices_[0]);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const cl_ulong num_classes = alpha_d.size(0);
    const cl_ulong num_sv = sv_d.size(0);
    const cl_ulong num_features = sv_d.size(1);

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const std::vector<std::size_t> block = { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const std::vector<std::size_t> grid = { static_cast<std::size_t>(std::ceil(static_cast<double>(num_features) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                            static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };

    device_ptr_type w_d{ { num_classes, num_features }, { PADDING_SIZE, PADDING_SIZE }, devices_[0] };

    detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::w_kernel), grid, block, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv);
    this->device_synchronize(devices_[0]);

    return w_d;
}

auto csvm::run_predict_kernel(const parameter &params, const device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const cl_ulong num_classes = alpha_d.size(0);
    const cl_ulong num_sv = sv_d.size(0);
    const cl_ulong num_predict_points = predict_points_d.size(0);
    const cl_ulong num_features = predict_points_d.size(1);

    device_ptr_type out_d{ { num_predict_points, num_classes }, { PADDING_SIZE, PADDING_SIZE }, devices_[0] };

    // define the block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const std::vector<std::size_t> block = { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };

    if (params.kernel_type == kernel_function_type::linear) {
        // define the grid sizes
        const std::vector<std::size_t> grid = { static_cast<std::size_t>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                                static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };

        detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::predict_kernel_linear), grid, block, out_d.get(), w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features);
    } else {
        // define the grid sizes
        const std::vector<std::size_t> grid = { static_cast<std::size_t>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                                static_cast<std::size_t>(std::ceil(static_cast<double>(num_sv) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::predict_kernel_polynomial), grid, block, out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.degree.value(), params.gamma.value(), params.coef0.value());
                break;
            case kernel_function_type::rbf:
                detail::run_kernel(devices_[0], devices_[0].get_kernel(detail::compute_kernel_name::predict_kernel_rbf), grid, block, out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.gamma.value());
                break;
        }
    }
    this->device_synchronize(devices_[0]);

    return out_d;
}

}  // namespace plssvm::opencl

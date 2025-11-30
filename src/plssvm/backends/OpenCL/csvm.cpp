/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/csvm.hpp"

#include "plssvm/backend_types.hpp"                        // plssvm::backend_type
#include "plssvm/backends/execution_range.hpp"             // plssvm::detail::{dim_type, execution_range}
#include "plssvm/backends/OpenCL/detail/jit_info.hpp"      // plssvm::opencl::detail::create_jit_report
#include "plssvm/backends/OpenCL/detail/kernel.hpp"        // plssvm::opencl::detail::{compute_kernel_name, kernel}
#include "plssvm/backends/OpenCL/detail/utility.hpp"       // PLSSVM_OPENCL_ERROR_CHECK, plssvm::opencl::detail::{get_contexts, create_command_queues, run_kernel, kernel_type_to_function_name, device_synchronize, get_opencl_target_version, get_driver_version}
#include "plssvm/backends/OpenCL/exceptions.hpp"           // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                            // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"             // plssvm::detail::{data_distribution, triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/logging/log_untracked.hpp"         // NOLINT: plssvm::detail::log_untracked
#include "plssvm/detail/logging/mpi_log_untracked.hpp"     // plssvm::detail::log_untracked
#include "plssvm/detail/memory_size.hpp"                   // plssvm::detail::memory_size
#include "plssvm/detail/operators.hpp"                     // operator overloads for std::vector (+ scalars)
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::{contains, check_local_memory_usage}
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                // plssvm::kernel_function_type
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/mpi/detail/information.hpp"               // plssvm::mpi::detail::gather_and_print_csvm_information
#include "plssvm/parameter.hpp"                            // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/shape.hpp"                                // plssvm::shape
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "CL/cl.h"           // CL_QUEUE_DEVICE, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE
                             // clGetCommandQueueInfo, clGetDeviceInfo, cl_device_id
#include "CL/cl_platform.h"  // cl_ulong

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::all_of
#include <chrono>     // std::chrono::{steady_clock, duration_cast}
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int32_t, std::uint16_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <limits>     // std::numeric_limits::max
#include <optional>   // std::optional
#include <string>     // std::string
#include <tuple>      // std::tie, std::get
#include <utility>    // std::pair, std::make_pair, std::move
#include <vector>     // std::vector

namespace plssvm::opencl {

csvm::csvm(const target_platform target) {
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

    // get all available OpenCL contexts for the current target including devices with respect to the requested target platform
    std::tie(contexts_, target_) = detail::get_contexts(target);

    // determine the used local memory and check whether it exceeds the maximum necessary value!
    // NOTE: must be before the JIT compilation. Otherwise, it may already fail there which leads to failing tests since the wrong error is created.
    // TODO: better solution: rewrite the OpenCL backend JIT compilation that all kernels are only JIT compiled right before they are used and not in the constructor itself!
    ::plssvm::detail::check_local_memory_usage(this->get_local_memory());

    // At this point, target_ may NEVER be target_platform::automatic!
    PLSSVM_ASSERT(target_ != target_platform::automatic, "At this point, the target platform must be determined and must NOT be automatic!");

    // at least one OpenCL context must be created
    if (contexts_.empty()) {
        throw backend_exception{ fmt::format("No OpenCL context for the target {} could be found!", target_) };
    }

    // create command_queues and JIT compile OpenCL kernels; compile all kernels for float and double
    detail::jit_info info{};
    std::tie(devices_, info) = detail::create_command_queues(comm_, contexts_, target_, params_.kernel_type);

    // for each context exactly ONE command queue must exist!
    PLSSVM_ASSERT(contexts_.size() == devices_.size(), "For each context ({}), exactly ONE command queue ({}) must exist!", contexts_.size(), devices_.size());

    std::vector<std::string> device_names{};
    device_names.reserve(devices_.size());
    std::vector<std::string> driver_versions{};
    driver_versions.reserve(comm_.size());

    if (comm_.size() > 1) {
        // use MPI rank specific command line output
        for (const queue_type &device : devices_) {
            device_names.push_back(detail::get_device_name(device));
            // get the target platform's driver version
            driver_versions.push_back(detail::get_driver_version(device));
        }

        mpi::detail::gather_and_print_csvm_information(comm_, plssvm::backend_type::opencl, target_, device_names, detail::create_jit_report(info));
    } else {
        // use more detailed single rank command line output
        plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                      comm_,
                                      "\nOpenCL kernel JIT compilation done in {}.\n",
                                      info.duration);
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "\nUsing OpenCL (target version: {}) as backend.\n",
                                      detail::get_opencl_target_version());
        if (target == target_platform::automatic) {
            plssvm::detail::log_untracked(verbosity_level::full,
                                          comm_,
                                          "Using {} as automatic target platform.\n",
                                          target_);
        }
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "Found {} OpenCL device(s) for the target platform {}:\n",
                                      devices_.size(),
                                      target_);

        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            device_names.push_back(detail::get_device_name(devices_[device]));
            plssvm::detail::log_untracked(verbosity_level::full,
                                          comm_,
                                          "  [{}, {}]\n",
                                          device,
                                          device_names.back());

            // get the target platform's driver version
            driver_versions.emplace_back(detail::get_driver_version(devices_[device]));
        }
    }

    plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "\n");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "opencl_target_version", detail::get_opencl_target_version() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "opencl_library", std::string{ PLSSVM_OPENCL_LIBRARY } }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "device_driver_version", driver_versions }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "backend", plssvm::backend_type::opencl }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", devices_.size() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_names }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "jit_compilation_time", info.duration }));

    // sanity checks for the number of the OpenCL kernels
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.size() == 17; }),
                  "Every command queue must have exactly 17 associated kernels!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::fill_kernel_float); }),
                  "The double device pointer fill kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::fill_kernel_double); }),
                  "The float device pointer fill kernel is missing!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::memset_kernel_float); }),
                  "The double device pointer memset kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::memset_kernel_double); }),
                  "The float device pointer memset kernel is missing!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::assemble_kernel_matrix_explicit); }),
                  "The explicit kernel matrix assembly device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::symm_kernel_explicit); }),
                  "The explicit BLAS SYMM device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::mirror_symm_kernel_explicit); }),
                  "The explicit BLAS mirror SYMM device kernel is missing!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas); }),
                  "The implicit kernel matrix assembly device kernel is missing!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::inplace_matrix_add_kernel); }),
                  "The inplace matrix add kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::inplace_matrix_scale_kernel); }),
                  "The inplace matrix scale kernel is missing!");

    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::w_kernel); }),
                  "The w_kernel device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_linear); }),
                  "The predict_kernel_linear device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_polynomial); }),
                  "The predict_kernel_polynomial device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_rbf); }),
                  "The predict_kernel_rbf device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_sigmoid); }),
                  "The predict_kernel_sigmoid device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_laplacian); }),
                  "The predict_kernel_laplacian device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return ::plssvm::detail::contains(queue.kernels, detail::compute_kernel_name::predict_kernel_chi_squared); }),
                  "The predict_kernel_chi_squared device kernel is missing!");
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

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    std::vector<::plssvm::detail::memory_size> res(contexts_.size());
    for (std::size_t device_id = 0; device_id < contexts_.size(); ++device_id) {
        // get device
        cl_device_id device = contexts_[device_id].device;

        // get device global memory size
        cl_ulong total_device_memory{};
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &total_device_memory, nullptr), "error obtaining device's global memory size")
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(total_device_memory) };
    }
    return res;
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    std::vector<::plssvm::detail::memory_size> res(contexts_.size());
    for (std::size_t device_id = 0; device_id < contexts_.size(); ++device_id) {
        // get device
        cl_device_id device = contexts_[device_id].device;

        // get maximum allocation size
        cl_ulong max_alloc_size{};
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, nullptr), "error obtaining device's maximum allocation size")
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(max_alloc_size) };
    }
    return res;
}

std::vector<std::optional<::plssvm::detail::memory_size>> csvm::get_local_memory() const {
    std::vector<std::optional<::plssvm::detail::memory_size>> res(contexts_.size());
    for (std::size_t device_id = 0; device_id < contexts_.size(); ++device_id) {
        // get device
        cl_device_id device = contexts_[device_id].device;

        // get the local memory size
        cl_ulong total_local_memory{};
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &total_local_memory, nullptr), "error obtaining device's local memory size")
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(total_local_memory) };
    }
    return res;
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < contexts_.size(), "Invalid device {} requested!", device_id);
    // get device
    cl_device_id device = contexts_[device_id].device;
    // get maximum work group size
    cl_ulong max_work_group_size{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &max_work_group_size, nullptr), "error obtaining device's global memory size")
    return static_cast<std::size_t>(max_work_group_size);
}

::plssvm::detail::dim_type csvm::get_max_grid_size([[maybe_unused]] const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < contexts_.size(), "Invalid device {} requested!", device_id);

    // TODO: replace with function if there will be one in the future
    // fallback to maximum theoretical value, may break at runtime!
    ::plssvm::detail::dim_type native_range{};
    const std::size_t max_int32 = std::numeric_limits<std::int32_t>::max();
    const std::size_t max_uint16 = std::numeric_limits<std::uint16_t>::max();
    if (target_ == target_platform::cpu) {
        native_range = ::plssvm::detail::dim_type{ max_int32, max_int32, max_int32 };
    } else {
        native_range = ::plssvm::detail::dim_type{ max_int32, max_uint16, max_uint16 };
    }
    return native_range;
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const cl_ulong num_rows_reduced = data_d.shape().x - 1;
    const cl_ulong num_features = data_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const cl_ulong device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const cl_ulong row_offset = data_distribution_->place_row_offset(device_id);

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, device };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range partial_grid to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), native_partial_grid, native_block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, grid_offset_x, grid_offset_y);
                break;
            case kernel_function_type::polynomial:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), native_partial_grid, native_block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, grid_offset_x, grid_offset_y, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), native_partial_grid, native_block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), native_partial_grid, native_block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), native_partial_grid, native_block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_explicit), native_partial_grid, native_block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "kernel_matrix", "kernel_matrix_assembly_kernel", duration }));

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const ::plssvm::detail::execution_range &mirror_exec, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const cl_ulong num_rhs = B_d.shape().x;
    const cl_ulong num_rows = B_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const cl_ulong device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const cl_ulong row_offset = data_distribution_->place_row_offset(device_id);

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range grid[i] to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::symm_kernel_explicit), native_partial_grid, native_block, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), grid_offset_x, grid_offset_y);
    }

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_mirror_block = detail::dim_type_to_native<2>(mirror_exec.block);

    for (const auto &[partial_grid, offsets] : mirror_exec.grids) {
        const cl_ulong num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

        if (num_mirror_rows > 0) {
            // convert execution range grid[i] to OpenCL's native std::vector
            const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_mirror_block;

            // cast offsets to OpenCL type
            const cl_ulong grid_offset_x = offsets.x;
            const cl_ulong grid_offset_y = offsets.y;

            detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::mirror_symm_kernel_explicit), native_partial_grid, native_mirror_block, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), grid_offset_x, grid_offset_y);
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "cg", "blas_level_3_times_kernel", duration }));
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const cl_ulong num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range partial_grid to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::inplace_matrix_add_kernel), native_partial_grid, native_block, num_rhs, lhs_d.get(), rhs_d.get(), grid_offset_x, grid_offset_y);
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
    const cl_ulong num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range partial_grid to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::inplace_matrix_scale_kernel), native_partial_grid, native_block, num_rhs, lhs_d.get(), scale, grid_offset_x, grid_offset_y);
    }
    detail::device_synchronize(device);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const cl_ulong num_rows_reduced = A_d.shape().x - 1;
    const cl_ulong num_features = A_d.shape().y;
    const cl_ulong num_classes = B_d.shape().x;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const cl_ulong device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const cl_ulong row_offset = data_distribution_->place_row_offset(device_id);

    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range partial_grid to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas), native_partial_grid, native_block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, grid_offset_x, grid_offset_y);
                break;
            case kernel_function_type::polynomial:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas), native_partial_grid, native_block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, grid_offset_x, grid_offset_y, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas), native_partial_grid, native_block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas), native_partial_grid, native_block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas), native_partial_grid, native_block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::assemble_kernel_matrix_implicit_blas), native_partial_grid, native_block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "cg", "blas_level_3_times_kernel", duration }));
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const cl_ulong num_classes = alpha_d.shape().x;
    const cl_ulong num_sv = alpha_d.shape().y;
    const cl_ulong device_specific_num_sv = sv_d.shape().x;
    const cl_ulong num_features = sv_d.shape().y;
    const queue_type &device = devices_[device_id];

    // get the offset of the data points this device is responsible for
    const cl_ulong sv_offset = data_distribution_->place_row_offset(device_id);

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range partial_grid to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::w_kernel), native_partial_grid, native_block, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, grid_offset_x, grid_offset_y);
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "predict_values", "w_kernel", duration }));

    return w_d;
}

auto csvm::run_predict_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const cl_ulong num_classes = alpha_d.shape().x;
    const cl_ulong num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const cl_ulong num_features = predict_points_d.shape().y;
    const cl_ulong num_sv = sv_or_w_d.shape().x;
    const queue_type &device = devices_[device_id];

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    // convert execution range block to OpenCL's native std::vector
    const std::vector<std::size_t> native_block = detail::dim_type_to_native<2>(exec.block);

    using namespace plssvm::operators;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        // convert execution range partial_grid to OpenCL's native std::vector
        const std::vector<std::size_t> native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        // cast offsets to OpenCL type
        const cl_ulong grid_offset_x = offsets.x;
        const cl_ulong grid_offset_y = offsets.y;

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::predict_kernel_linear), native_partial_grid, native_block, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, grid_offset_x, grid_offset_y);
                break;
            case kernel_function_type::polynomial:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::predict_kernel_polynomial), native_partial_grid, native_block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, grid_offset_x, grid_offset_y, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::predict_kernel_rbf), native_partial_grid, native_block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::predict_kernel_sigmoid), native_partial_grid, native_block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::predict_kernel_laplacian), native_partial_grid, native_block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::run_kernel(device, device.get_kernel(detail::compute_kernel_name::predict_kernel_chi_squared), native_partial_grid, native_block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, grid_offset_x, grid_offset_y, std::get<real_type>(params.gamma));
                break;
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "predict_values", "predict_kernel", duration }));

    return out_d;
}

}  // namespace plssvm::opencl

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/AdaptiveCpp/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                              // plssvm::backend_type
#include "plssvm/backends/execution_range.hpp"                                                   // plssvm::detail::{dim_type, execution_range}
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/device_ptr.hpp"                                // plssvm::adaptivecpp::detail::::device_ptr
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/queue_impl.hpp"                                // plssvm::adaptivecpp::detail::queue (PImpl implementation)
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/utility.hpp"                                   // plssvm::adaptivecpp::detail::{get_device_list, device_synchronize, get_device_name, get_adaptivecpp_version_short, get_adaptivecpp_version}
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"                                        // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/exceptions.hpp"                                                   // plssvm::adaptivecpp::backend_exception
#include "plssvm/backends/SYCL/implementation_types.hpp"                                         // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel/cg_explicit/basic/blas.hpp"                                // plssvm::sycl::detail::basic::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/SYCL/kernel/cg_explicit/basic/kernel_matrix_assembly.hpp"              // plssvm::sycl::detail::basic::device_kernel_assembly
#include "plssvm/backends/SYCL/kernel/cg_explicit/hierarchical/blas.hpp"                         // plssvm::sycl::detail::hierarchical::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/SYCL/kernel/cg_explicit/hierarchical/kernel_matrix_assembly.hpp"       // plssvm::sycl::detail::hierarchical::device_kernel_assembly
#include "plssvm/backends/SYCL/kernel/cg_explicit/scoped/blas.hpp"                               // plssvm::sycl::detail::scoped::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/SYCL/kernel/cg_explicit/scoped/kernel_matrix_assembly.hpp"             // plssvm::sycl::detail::scoped::device_kernel_assembly
#include "plssvm/backends/SYCL/kernel/cg_explicit/work_group/blas.hpp"                           // plssvm::sycl::detail::work_group::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/SYCL/kernel/cg_explicit/work_group/kernel_matrix_assembly.hpp"         // plssvm::sycl::detail::work_group::device_kernel_assembly
#include "plssvm/backends/SYCL/kernel/cg_implicit/basic/kernel_matrix_assembly_blas.hpp"         // plssvm::sycl::detail::basic::device_kernel_assembly_symm
#include "plssvm/backends/SYCL/kernel/cg_implicit/hierarchical/kernel_matrix_assembly_blas.hpp"  // plssvm::sycl::detail::hierarchical::device_kernel_assembly_symm
#include "plssvm/backends/SYCL/kernel/cg_implicit/scoped/kernel_matrix_assembly_blas.hpp"        // plssvm::sycl::detail::scoped::device_kernel_assembly_symm
#include "plssvm/backends/SYCL/kernel/cg_implicit/work_group/kernel_matrix_assembly_blas.hpp"    // plssvm::sycl::detail::work_group::device_kernel_assembly_symm
#include "plssvm/backends/SYCL/kernel/predict/basic/predict_kernel.hpp"                          // plssvm::sycl::detail::basic::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/backends/SYCL/kernel/predict/hierarchical/predict_kernel.hpp"                   // plssvm::sycl::detail::hierarchical::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/backends/SYCL/kernel/predict/scoped/predict_kernel.hpp"                         // plssvm::sycl::detail::scoped::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/backends/SYCL/kernel/predict/work_group/predict_kernel.hpp"                     // plssvm::sycl::detail::work_group::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                                  // plssvm::real_type
#include "plssvm/detail/assert.hpp"                                                              // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"                                                   // plssvm::detail::{data_distribution, triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/logging/log.hpp"                                                         // plssvm::detail::log
#include "plssvm/detail/logging/log_untracked.hpp"                                               // plssvm::detail::log_untracked
#include "plssvm/detail/logging/mpi_log_untracked.hpp"                                           // plssvm::detail::log_untracked
#include "plssvm/detail/memory_size.hpp"                                                         // plssvm::detail::memory_size
#include "plssvm/detail/tracking/performance_tracker.hpp"                                        // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/utility.hpp"                                                             // plssvm::detail::get_system_memory
#include "plssvm/kernel_function_types.hpp"                                                      // plssvm::kernel_type
#include "plssvm/mpi/communicator.hpp"                                                           // plssvm::mpi::communicator
#include "plssvm/mpi/detail/information.hpp"                                                     // plssvm::mpi::detail::gather_and_print_csvm_information
#include "plssvm/parameter.hpp"                                                                  // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/shape.hpp"                                                                      // plssvm::shape
#include "plssvm/target_platforms.hpp"                                                           // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                                           // plssvm::verbosity_level

#include "sycl/sycl.hpp"  // sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include "fmt/color.h"   // fmt::fg, fmt::color::orange
#include "fmt/format.h"  // fmt::format

#include <chrono>     // std::chrono::{steady_clock, duration_cast}
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int32_t, std::uint16_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <limits>     // std::numeric_limits::max
#include <optional>   // std::optional
#include <string>     // std::string
#include <tuple>      // std::tie, std::get
#include <utility>    // std::forward
#include <vector>     // std::vector

namespace {

/**
 * @brief Run the kernel functor on the given device.
 * @tparam KernelFunctor the type of the kernel functor to run
 * @tparam QueueType the type of the SYCL queue to run the kernel on
 * @tparam Args the types of the parameters necessary for the specific kernel functor
 * @param[in] device the SYCL queue to run the kernel on
 * @param[in] partial_grid the number of work-groups in each dimension of the execution grid
 * @param[in] block the number of work-items in each dimension per work-group
 * @param[in] args the parameters necessary for the specific kernel functor
 */
template <typename KernelFunctor, typename QueueType, typename... Args>
void run_kernel_functor(const QueueType &device, const plssvm::detail::dim_type partial_grid, const plssvm::detail::dim_type block, Args &&...args) {
    constexpr plssvm::sycl::data_parallel_kernel data_parallel_kernel_type = KernelFunctor::data_parallel_kernel_type;

    if constexpr (data_parallel_kernel_type == plssvm::sycl::data_parallel_kernel::basic) {
        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            cgh.parallel_for(plssvm::adaptivecpp::detail::get_execution_range<plssvm::sycl::data_parallel_kernel::basic>(partial_grid, block),
                             KernelFunctor{ std::forward<Args>(args)... });
        });
    } else if constexpr (data_parallel_kernel_type == plssvm::sycl::data_parallel_kernel::work_group) {
        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            cgh.parallel_for(plssvm::adaptivecpp::detail::get_execution_range<plssvm::sycl::data_parallel_kernel::work_group>(partial_grid, block),
                             KernelFunctor{ cgh, std::forward<Args>(args)... });
        });
    } else if constexpr (data_parallel_kernel_type == plssvm::sycl::data_parallel_kernel::hierarchical) {
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            const auto exec_range = plssvm::adaptivecpp::detail::get_execution_range<plssvm::sycl::data_parallel_kernel::hierarchical>(partial_grid, block);
            cgh.parallel_for_work_group(exec_range.get_global_range(), exec_range.get_local_range(), KernelFunctor{ std::forward<Args>(args)... });
        });
#else
        throw plssvm::adaptivecpp::backend_exception{ "Support for sycl::data_parallel_kernel::hierarchical was disabled!" };
#endif
    } else if constexpr (data_parallel_kernel_type == plssvm::sycl::data_parallel_kernel::scoped) {
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            const auto exec_range = plssvm::adaptivecpp::detail::get_execution_range<plssvm::sycl::data_parallel_kernel::scoped>(partial_grid, block);
            cgh.parallel(exec_range.get_global_range(), exec_range.get_local_range(), KernelFunctor{ std::forward<Args>(args)... });
        });
#else
        throw plssvm::adaptivecpp::backend_exception{ "Support for sycl::data_parallel_kernel::scoped was disabled!" };
#endif
    } else {
        static_assert(::plssvm::detail::always_false_v<Args...>, "Unsupported kernel function!");
    }
}

/**
 * @brief Dispatch the kernel functor to the correct kernel function type.
 * @tparam KernelFunctor the type of the kernel functor to run
 * @tparam Args the types of the parameters necessary for the specific kernel functor; stored in a `std::tuple`
 * @param[in] params the parameters used to determine the kernel function type
 * @param[in] args the parameters necessary for the specific kernel functor
 */
template <template <plssvm::kernel_function_type, typename...> typename KernelFunctor, typename... Args>
void dispatch_kernel_functor(const plssvm::parameter &params, Args &&...args) {
    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            run_kernel_functor<KernelFunctor<plssvm::kernel_function_type::linear>>(std::forward<Args>(args)...);
            break;
        case plssvm::kernel_function_type::polynomial:
            run_kernel_functor<KernelFunctor<plssvm::kernel_function_type::polynomial, int, plssvm::real_type, plssvm::real_type>>(std::forward<Args>(args)..., params.degree, std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::rbf:
            run_kernel_functor<KernelFunctor<plssvm::kernel_function_type::rbf, plssvm::real_type>>(std::forward<Args>(args)..., std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::sigmoid:
            run_kernel_functor<KernelFunctor<plssvm::kernel_function_type::sigmoid, plssvm::real_type, plssvm::real_type>>(std::forward<Args>(args)..., std::get<plssvm::real_type>(params.gamma), params.coef0);
            break;
        case plssvm::kernel_function_type::laplacian:
            run_kernel_functor<KernelFunctor<plssvm::kernel_function_type::laplacian, plssvm::real_type>>(std::forward<Args>(args)..., std::get<plssvm::real_type>(params.gamma));
            break;
        case plssvm::kernel_function_type::chi_squared:
            run_kernel_functor<KernelFunctor<plssvm::kernel_function_type::chi_squared, plssvm::real_type>>(std::forward<Args>(args)..., std::get<plssvm::real_type>(params.gamma));
            break;
    }
}

/**
 * @brief Dispatch the kernel functor to the correct target platform.
 * @tparam KernelFunctor the type of the kernel functor to run
 * @tparam Args the types of the parameters necessary for the specific kernel functor; stored in a `std::tuple`
 * @param[in] args the parameters necessary for the specific kernel functor
 */
template <typename KernelFunctor, typename... Args>
void dispatch_kernel_functor(Args &&...args) {
    run_kernel_functor<KernelFunctor>(std::forward<Args>(args)...);
}

}  // namespace

namespace plssvm::adaptivecpp {

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

    // get all available devices wrt the requested target platform
    std::tie(devices_, target_) = detail::get_device_list(target);

    // At this point, target_ may NEVER be target_platform::automatic!
    PLSSVM_ASSERT(target_ != target_platform::automatic, "At this point, the target platform must be determined and must NOT be automatic!");

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", target_) };
    }

    // set the correct data parallel kernel if "automatic" has been provided
    if (data_parallel_kernel_type_ == sycl::data_parallel_kernel::automatic) {
        // always use work_group for AdaptiveCpp
        data_parallel_kernel_type_ = sycl::data_parallel_kernel::work_group;
        if (target_ == target_platform::cpu) {
#if !defined(__ACPP_USE_ACCELERATED_CPU__) && defined(__ACPP_ENABLE_OMPHOST_TARGET__)
            plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                                          "WARNING: the AdaptiveCpp automatic target for the CPU is set to work_group, but AdaptiveCpp hasn't been build with the \"omp.accelerated\" compilation flow resulting in major performance losses!\n");
#endif
        }
    }

    std::vector<std::string> device_names{};
    device_names.reserve(devices_.size());

    if (comm_.size() > 1) {
        // use MPI rank specific command line output
        for (const queue_type &device : devices_) {
            device_names.push_back(detail::get_device_name(device));
        }

        mpi::detail::gather_and_print_csvm_information(comm_, plssvm::backend_type::sycl, target_, device_names, fmt::format("{}", data_parallel_kernel_type_));
    } else {
        // use more detailed single rank command line output
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "\nUsing AdaptiveCpp ({}; {}) as SYCL backend with the data parallel kernel \"{}\".\n",
                                      detail::get_adaptivecpp_version_short(),
                                      PLSSVM_ACPP_TARGETS,
                                      data_parallel_kernel_type_);
        if (target == target_platform::automatic) {
            plssvm::detail::log_untracked(verbosity_level::full,
                                          comm_,
                                          "Using {} as automatic target platform.\n",
                                          target_);
        }
        plssvm::detail::log_untracked(verbosity_level::full,
                                      comm_,
                                      "Found {} SYCL device(s) for the target platform {}:\n",
                                      devices_.size(),
                                      target_);

        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            device_names.push_back(detail::get_device_name(devices_[device]));
            plssvm::detail::log_untracked(verbosity_level::full,
                                          comm_,
                                          "  [{}, {}]\n",
                                          device,
                                          device_names.back());
        }
    }

    plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::timing,
                                  comm_,
                                  "\n");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "adaptivecpp_version", detail::get_adaptivecpp_version() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "backend", plssvm::backend_type::sycl }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "sycl_implementation_type", plssvm::sycl::implementation_type::adaptivecpp }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "sycl_data_parallel_kernel", data_parallel_kernel_type_ }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", devices_.size() }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_names }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "acpp_targets", PLSSVM_ACPP_TARGETS }));
}

csvm::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (const queue_type &q : devices_) {
            detail::device_synchronize(q);
        }
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        std::terminate();
    }
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        const ::plssvm::detail::memory_size adaptivecpp_global_mem_size{ static_cast<unsigned long long>(devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::global_mem_size>()) };
        if (target_ == target_platform::cpu) {
            plssvm::detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                                          "WARNING: the returned 'global_mem_size' for AdaptiveCpp targeting the CPU device {} is nonsensical ('std::numeric_limits<std::size_t>::max()'). Using 'get_system_memory()' instead.\n",
                                          device_id);
            res[device_id] = std::min(adaptivecpp_global_mem_size, ::plssvm::detail::get_system_memory());
        } else {
            res[device_id] = adaptivecpp_global_mem_size;
        }
    }
    return res;
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_mem_alloc_size>()) };
    }
    return res;
}

std::vector<std::optional<::plssvm::detail::memory_size>> csvm::get_local_memory() const {
    std::vector<std::optional<::plssvm::detail::memory_size>> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::local_mem_size>()) };
    }
    return res;
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    return devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_work_group_size>();
}

::plssvm::detail::dim_type csvm::get_max_grid_size([[maybe_unused]] const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);

    // TODO: replace with function if there will be one in the future
    // fallback to maximum theoretical value, may break at runtime!
    ::sycl::id<3> native_range{};
    const std::size_t max_int32 = std::numeric_limits<std::int32_t>::max();
    const std::size_t max_uint16 = std::numeric_limits<std::uint16_t>::max();
    if (target_ == target_platform::cpu) {
        native_range = ::sycl::id<3>{ max_int32, max_int32, max_int32 };
    } else {
        native_range = ::sycl::id<3>{ max_int32, max_uint16, max_uint16 };
    }

    // note: account for SYCL's different iteration range!
    return { native_range[2], native_range[1], native_range[0] };
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const std::size_t num_rows_reduced = data_d.shape().x - 1;
    const std::size_t num_features = data_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, device };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        switch (data_parallel_kernel_type_) {
            case sycl::data_parallel_kernel::automatic:
                throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
                break;
            case sycl::data_parallel_kernel::basic:
                dispatch_kernel_functor<sycl::detail::basic::device_kernel_assembly>(params, device, partial_grid, exec.block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::work_group:
                dispatch_kernel_functor<sycl::detail::work_group::device_kernel_assembly>(params, device, partial_grid, exec.block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::hierarchical:
                dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_assembly>(params, device, partial_grid, exec.block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::scoped:
                dispatch_kernel_functor<sycl::detail::scoped::device_kernel_assembly>(params, device, partial_grid, exec.block, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.y, offsets.x);
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "kernel_matrix", "kernel_matrix_assembly_kernel", duration }));

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const ::plssvm::detail::execution_range &mirror_exec, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const std::size_t num_rhs = B_d.shape().x;
    const std::size_t num_rows = B_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        switch (data_parallel_kernel_type_) {
            case sycl::data_parallel_kernel::automatic:
                throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
            case sycl::data_parallel_kernel::basic:
                dispatch_kernel_functor<sycl::detail::basic::device_kernel_symm>(device, partial_grid, exec.block, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::work_group:
                dispatch_kernel_functor<sycl::detail::work_group::device_kernel_symm>(device, partial_grid, exec.block, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::hierarchical:
                dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_symm>(device, partial_grid, exec.block, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::scoped:
                dispatch_kernel_functor<sycl::detail::scoped::device_kernel_symm>(device, partial_grid, exec.block, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
        }
    }

    for (const auto &[partial_grid, offsets] : mirror_exec.grids) {
        const unsigned long long num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

        if (num_mirror_rows > 0) {
            switch (data_parallel_kernel_type_) {
                case sycl::data_parallel_kernel::automatic:
                    throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
                case sycl::data_parallel_kernel::basic:
                    dispatch_kernel_functor<sycl::detail::basic::device_kernel_symm_mirror>(device, partial_grid, exec.block, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::work_group:
                    dispatch_kernel_functor<sycl::detail::work_group::device_kernel_symm_mirror>(device, partial_grid, exec.block, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::hierarchical:
                    dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_symm_mirror>(device, partial_grid, exec.block, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::scoped:
                    dispatch_kernel_functor<sycl::detail::scoped::device_kernel_symm_mirror>(device, partial_grid, exec.block, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.y, offsets.x);
            }
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "cg", "blas_level_3_times_kernel", duration }));
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const std::size_t num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    for (const auto &[partial_grid, offsets] : exec.grids) {
        switch (data_parallel_kernel_type_) {
            case sycl::data_parallel_kernel::automatic:
                throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
            case sycl::data_parallel_kernel::basic:
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    cgh.parallel_for(detail::get_execution_range<sycl::data_parallel_kernel::basic>(partial_grid_ref, exec.block),
                                     sycl::detail::basic::device_kernel_inplace_matrix_add{ num_rhs, lhs_d.get(), rhs_d.get(), offsets_ref.y, offsets_ref.x });
                });
                break;
            case sycl::data_parallel_kernel::work_group:
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    cgh.parallel_for(detail::get_execution_range<sycl::data_parallel_kernel::work_group>(partial_grid_ref, exec.block),
                                     sycl::detail::work_group::device_kernel_inplace_matrix_add{ num_rhs, lhs_d.get(), rhs_d.get(), offsets_ref.y, offsets_ref.x });
                });
                break;
            case sycl::data_parallel_kernel::hierarchical:
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    const auto exec_range = detail::get_execution_range<sycl::data_parallel_kernel::hierarchical>(partial_grid_ref, exec.block);
                    cgh.parallel_for_work_group(exec_range.get_global_range(), exec_range.get_local_range(), sycl::detail::hierarchical::device_kernel_inplace_matrix_add{ num_rhs, lhs_d.get(), rhs_d.get(), offsets_ref.y, offsets_ref.x });
                });
#else
                throw backend_exception{ "Support for sycl::data_parallel_kernel::hierarchical was disabled!" };
#endif
                break;
            case sycl::data_parallel_kernel::scoped:
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    const auto exec_range = detail::get_execution_range<sycl::data_parallel_kernel::scoped>(partial_grid_ref, exec.block);
                    cgh.parallel(exec_range.get_global_range(), exec_range.get_local_range(), sycl::detail::scoped::device_kernel_inplace_matrix_add{ num_rhs, lhs_d.get(), rhs_d.get(), offsets_ref.y, offsets_ref.x });
                });
#else
                throw backend_exception{ "Support for sycl::data_parallel_kernel::scoped was disabled!" };
#endif
                break;
        }
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
    const std::size_t num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    for (const auto &[partial_grid, offsets] : exec.grids) {
        switch (data_parallel_kernel_type_) {
            case sycl::data_parallel_kernel::automatic:
                throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
            case sycl::data_parallel_kernel::basic:
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    cgh.parallel_for(detail::get_execution_range<sycl::data_parallel_kernel::basic>(partial_grid_ref, exec.block),
                                     sycl::detail::basic::device_kernel_inplace_matrix_scale{ num_rhs, lhs_d.get(), scale, offsets_ref.y, offsets_ref.x });
                });
                break;
            case sycl::data_parallel_kernel::work_group:
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    cgh.parallel_for(detail::get_execution_range<sycl::data_parallel_kernel::work_group>(partial_grid_ref, exec.block),
                                     sycl::detail::work_group::device_kernel_inplace_matrix_scale{ num_rhs, lhs_d.get(), scale, offsets_ref.y, offsets_ref.x });
                });
                break;
            case sycl::data_parallel_kernel::hierarchical:
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    const auto exec_range = detail::get_execution_range<sycl::data_parallel_kernel::hierarchical>(partial_grid_ref, exec.block);
                    cgh.parallel_for_work_group(exec_range.get_global_range(), exec_range.get_local_range(), sycl::detail::hierarchical::device_kernel_inplace_matrix_scale{ num_rhs, lhs_d.get(), scale, offsets_ref.y, offsets_ref.x });
                });
#else
                throw backend_exception{ "Support for sycl::data_parallel_kernel::hierarchical was disabled!" };
#endif
                break;
            case sycl::data_parallel_kernel::scoped:
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
                device.impl->sycl_queue.submit([&, &partial_grid_ref = partial_grid, &offsets_ref = offsets](::sycl::handler &cgh) {
                    const auto exec_range = detail::get_execution_range<sycl::data_parallel_kernel::scoped>(partial_grid_ref, exec.block);
                    cgh.parallel(exec_range.get_global_range(), exec_range.get_local_range(), sycl::detail::scoped::device_kernel_inplace_matrix_scale{ num_rhs, lhs_d.get(), scale, offsets_ref.y, offsets_ref.x });
                });
#else
                throw backend_exception{ "Support for sycl::data_parallel_kernel::scoped was disabled!" };
#endif
                break;
        }
    }
    detail::device_synchronize(device);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const std::size_t num_rows_reduced = A_d.shape().x - 1;
    const std::size_t num_features = A_d.shape().y;
    const std::size_t num_classes = B_d.shape().x;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        switch (data_parallel_kernel_type_) {
            case sycl::data_parallel_kernel::automatic:
                throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
                break;
            case sycl::data_parallel_kernel::basic:
                dispatch_kernel_functor<sycl::detail::basic::device_kernel_assembly_symm>(params, device, partial_grid, exec.block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::work_group:
                dispatch_kernel_functor<sycl::detail::work_group::device_kernel_assembly_symm>(params, device, partial_grid, exec.block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::hierarchical:
                dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_assembly_symm>(params, device, partial_grid, exec.block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::scoped:
                dispatch_kernel_functor<sycl::detail::scoped::device_kernel_assembly_symm>(params, device, partial_grid, exec.block, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.y, offsets.x);
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
    const std::size_t num_classes = alpha_d.shape().x;
    const std::size_t num_sv = alpha_d.shape().y;
    const std::size_t device_specific_num_sv = sv_d.shape().x;
    const std::size_t num_features = sv_d.shape().y;
    const queue_type &device = devices_[device_id];

    // get the offset of the data points this device is responsible for
    const std::size_t sv_offset = data_distribution_->place_row_offset(device_id);

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        switch (data_parallel_kernel_type_) {
            case sycl::data_parallel_kernel::automatic:
                throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
            case sycl::data_parallel_kernel::basic:
                dispatch_kernel_functor<sycl::detail::basic::device_kernel_w_linear>(device, partial_grid, exec.block, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::work_group:
                dispatch_kernel_functor<sycl::detail::work_group::device_kernel_w_linear>(device, partial_grid, exec.block, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::hierarchical:
                dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_w_linear>(device, partial_grid, exec.block, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.y, offsets.x);
                break;
            case sycl::data_parallel_kernel::scoped:
                dispatch_kernel_functor<sycl::detail::scoped::device_kernel_w_linear>(device, partial_grid, exec.block, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.y, offsets.x);
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "predict_values", "w_kernel", duration }));

    return w_d;
}

auto csvm::run_predict_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.shape().x;
    const std::size_t num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const std::size_t num_features = predict_points_d.shape().y;
    const std::size_t num_sv = sv_or_w_d.shape().x;
    const queue_type &device = devices_[device_id];

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    const auto start = std::chrono::steady_clock::now();
    for (const auto &[partial_grid, offsets] : exec.grids) {
        if (params.kernel_type == kernel_function_type::linear) {
            switch (data_parallel_kernel_type_) {
                case sycl::data_parallel_kernel::automatic:
                    throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
                case sycl::data_parallel_kernel::basic:
                    dispatch_kernel_functor<sycl::detail::basic::device_kernel_predict_linear>(device, partial_grid, exec.block, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::work_group:
                    dispatch_kernel_functor<sycl::detail::work_group::device_kernel_predict_linear>(device, partial_grid, exec.block, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::hierarchical:
                    dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_predict_linear>(device, partial_grid, exec.block, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::scoped:
                    dispatch_kernel_functor<sycl::detail::scoped::device_kernel_predict_linear>(device, partial_grid, exec.block, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.y, offsets.x);
            }
        } else {
            switch (data_parallel_kernel_type_) {
                case sycl::data_parallel_kernel::automatic:
                    throw backend_exception{ "Can't determine the sycl::data_parallel_kernel!" };
                case sycl::data_parallel_kernel::basic:
                    dispatch_kernel_functor<sycl::detail::basic::device_kernel_predict>(params, device, partial_grid, exec.block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::work_group:
                    dispatch_kernel_functor<sycl::detail::work_group::device_kernel_predict>(params, device, partial_grid, exec.block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::hierarchical:
                    dispatch_kernel_functor<sycl::detail::hierarchical::device_kernel_predict>(params, device, partial_grid, exec.block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.y, offsets.x);
                    break;
                case sycl::data_parallel_kernel::scoped:
                    dispatch_kernel_functor<sycl::detail::scoped::device_kernel_predict>(params, device, partial_grid, exec.block, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.y, offsets.x);
            }
        }
    }
    detail::device_synchronize(device);
    const auto end = std::chrono::steady_clock::now();
    [[maybe_unused]] const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "predict_values", "predict_kernel", duration }));

    return out_d;
}

}  // namespace plssvm::adaptivecpp

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                  // plssvm::backend_type
#include "plssvm/backends/SYCL/cg_explicit/blas_basic.hpp"                           // plssvm::sycl::detail::basic::{device_kernel_gemm, device_kernel_symm}
#include "plssvm/backends/SYCL/cg_explicit/blas_hierarchical.hpp"                    // plssvm::sycl::detail::hierarchical::{device_kernel_gemm, device_kernel_symm}
#include "plssvm/backends/SYCL/cg_explicit/blas_work_group.hpp"                      // plssvm::sycl::detail::work_group::{device_kernel_gemm, device_kernel_symm}
#include "plssvm/backends/SYCL/cg_explicit/kernel_matrix_assembly_basic.hpp"         // plssvm::sycl::detail::basic::{device_kernel_assembly_linear, device_kernel_assembly_polynomial, device_kernel_assembly_rbf}
#include "plssvm/backends/SYCL/cg_explicit/kernel_matrix_assembly_hierarchical.hpp"  // plssvm::sycl::detail::hierarchical::{device_kernel_assembly_linear, device_kernel_assembly_polynomial, device_kernel_assembly_rbf}
#include "plssvm/backends/SYCL/cg_explicit/kernel_matrix_assembly_work_group.hpp"    // plssvm::sycl::detail::work_group::{device_kernel_assembly_linear, device_kernel_assembly_polynomial, device_kernel_assembly_rbf}
#include "plssvm/backends/SYCL/cg_implicit/kernel_matrix_assembly_blas.hpp"          // plssvm::sycl::{device_kernel_assembly_linear_symm, device_kernel_assembly_polynomial_symm, device_kernel_assembly_rbf_symm}
#include "plssvm/backends/SYCL/detail/utility.hpp"                                   // plssvm::sycl::detail::calculate_execution_range
#include "plssvm/backends/SYCL/DPCPP/detail/device_ptr.hpp"                          // plssvm::dpcpp::detail::::device_ptr
#include "plssvm/backends/SYCL/DPCPP/detail/queue_impl.hpp"                          // plssvm::dpcpp::detail::queue (PImpl implementation)
#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"                             // plssvm::dpcpp::detail::{get_device_list, device_synchronize, get_dpcpp_version}
#include "plssvm/backends/SYCL/exceptions.hpp"                                       // plssvm::dpcpp::backend_exception
#include "plssvm/backends/SYCL/predict_kernel.hpp"                                   // plssvm::sycl::detail::{kernel_w, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/constants.hpp"                                                      // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                                                  // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                                                 // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                             // plssvm::detail::memory_size
#include "plssvm/detail/performance_tracker.hpp"                                     // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                                          // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                                          // plssvm::kernel_type
#include "plssvm/parameter.hpp"                                                      // plssvm::parameter
#include "plssvm/shape.hpp"                                                          // plssvm::shape
#include "plssvm/target_platforms.hpp"                                               // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                               // plssvm::verbosity_level

#include "sycl/sycl.hpp"  // sycl::queue, sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <tuple>      // std::tie
#include <vector>     // std::vector

namespace plssvm::dpcpp {

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

    // get all available devices wrt the requested target platform
    std::tie(devices_, target_) = detail::get_device_list(target);

    // currently only single GPU execution is supported
    if (devices_.size() > 1) {
        plssvm::detail::log(verbosity_level::full | verbosity_level::warning,
                            "WARNING: found {} devices, but currently only single GPU execution is supported. Continuing only with device 0!\n",
                            devices_.size());
        devices_.resize(1);
    }

    // set correct kernel invocation type if "automatic" has been provided
    if (invocation_type_ == sycl::kernel_invocation_type::automatic) {
        // always use work_group for DPC++
        invocation_type_ = sycl::kernel_invocation_type::work_group;
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing DPC++ ({}; {}) as SYCL backend with the kernel invocation type \"{}\" for the svm_kernel.\n",
                        plssvm::detail::tracking_entry{ "dependencies", "dpcpp_version", detail::get_dpcpp_version() },
                        plssvm::detail::tracking_entry{ "dependencies", "dpcpp_timestamp_version", detail::get_dpcpp_timestamp_version() },
                        plssvm::detail::tracking_entry{ "backend", "sycl_kernel_invocation_type", invocation_type_ });
    if (target == target_platform::automatic) {
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::sycl }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "sycl_implementation_type", plssvm::sycl::implementation_type::dpcpp }));

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", target_) };
    }

    // print found SYCL devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} SYCL device(s) for the target platform {}:\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking_entry{ "backend", "target_platform", target_ });
    std::vector<std::string> device_names;
    device_names.reserve(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::string device_name = devices_[device].impl->sycl_queue.get_device().template get_info<::sycl::info::device::name>();
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}]\n",
                            device,
                            device_name);
        device_names.emplace_back(device_name);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "device", device_names }));
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (const queue_type &q : devices_) {
            device_synchronize(q);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what() << std::endl;
        std::terminate();
    }
}

::plssvm::detail::memory_size csvm::get_device_memory() const {
    return ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[0].impl->sycl_queue.get_device().get_info<::sycl::info::device::global_mem_size>()) };
}

::plssvm::detail::memory_size csvm::get_max_mem_alloc_size() const {
    return ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[0].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_mem_alloc_size>()) };
}

std::size_t csvm::get_max_work_group_size() const {
    return devices_[0].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_work_group_size>();
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const unsigned long long num_rows_reduced = data_d.shape().x - 1;
    const unsigned long long num_features = data_d.shape().y;

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE_OLD.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::nd_range<2> execution_range = sycl::detail::calculate_execution_range(::sycl::range<2>{ num_rows_reduced, num_rows_reduced }, invocation_type_);

#if defined(PLSSVM_USE_GEMM)
    device_ptr_type kernel_matrix_d{ (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE), devices_[0] };  // store full matrix
#else
    device_ptr_type kernel_matrix_d{ (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2, devices_[0] };  // only explicitly store the upper triangular matrix
#endif
    kernel_matrix_d.memset(0);
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (invocation_type_) {
        case sycl::kernel_invocation_type::automatic:
            throw exception{ "Can't determine the sycl::kernel_invocation_type!" };
        case sycl::kernel_invocation_type::basic:
            // basic data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                switch (params.kernel_type.value()) {
                    case kernel_function_type::linear:
                        cgh.parallel_for(::sycl::range<2>{ num_rows_reduced, num_rows_reduced }, sycl::detail::basic::device_kernel_assembly_linear{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor });
                        break;
                    case kernel_function_type::polynomial:
                        cgh.parallel_for(::sycl::range<2>{ num_rows_reduced, num_rows_reduced }, sycl::detail::basic::device_kernel_assembly_polynomial{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value() });
                        break;
                    case kernel_function_type::rbf:
                        cgh.parallel_for(::sycl::range<2>{ num_rows_reduced, num_rows_reduced }, sycl::detail::basic::device_kernel_assembly_rbf{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value() });
                        break;
                }
            });
            break;
        case sycl::kernel_invocation_type::work_group:
            // work-group data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                switch (params.kernel_type.value()) {
                    case kernel_function_type::linear:
                        cgh.parallel_for(execution_range, sycl::detail::work_group::device_kernel_assembly_linear{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor });
                        break;
                    case kernel_function_type::polynomial:
                        cgh.parallel_for(execution_range, sycl::detail::work_group::device_kernel_assembly_polynomial{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value() });
                        break;
                    case kernel_function_type::rbf:
                        cgh.parallel_for(execution_range, sycl::detail::work_group::device_kernel_assembly_rbf{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value() });
                        break;
                }
            });
            break;
        case sycl::kernel_invocation_type::hierarchical:
            // hierarchical data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                switch (params.kernel_type.value()) {
                    case kernel_function_type::linear:
                        cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical::device_kernel_assembly_linear{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor });
                        break;
                    case kernel_function_type::polynomial:
                        cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical::device_kernel_assembly_polynomial{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value() });
                        break;
                    case kernel_function_type::rbf:
                        cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical::device_kernel_assembly_rbf{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value() });
                        break;
                }
            });
            break;
        case sycl::kernel_invocation_type::scoped:
            throw exception{ "Can't use the sycl::kernel_invocation_type::scoped together with DPC++!" };
    }
    detail::device_synchronize(devices_[0]);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const unsigned long long num_rhs = B_d.shape().x;
    const unsigned long long num_rows = B_d.shape().y;

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE_OLD.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::nd_range<2> execution_range = sycl::detail::calculate_execution_range(::sycl::range<2>{ num_rows, num_rhs }, invocation_type_);

#if defined(PLSSVM_USE_GEMM)
    switch (invocation_type_) {
        case sycl::kernel_invocation_type::automatic:
            throw exception{ "Can't determine the sycl::kernel_invocation_type!" };
        case sycl::kernel_invocation_type::basic:
            // basic data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(::sycl::range<2>{ num_rows, num_rhs }, sycl::detail::basic::device_kernel_gemm{ num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
            break;
        case sycl::kernel_invocation_type::work_group:
            // work-group data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::work_group::device_kernel_gemm{ cgh, num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
            break;
        case sycl::kernel_invocation_type::hierarchical:
            // hierarchical data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical::device_kernel_gemm{ num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
            break;
        case sycl::kernel_invocation_type::scoped:
            throw exception{ "Can't use the sycl::kernel_invocation_type::scoped together with DPC++!" };
    }
#else
    switch (invocation_type_) {
        case sycl::kernel_invocation_type::automatic:
            throw exception{ "Can't determine the sycl::kernel_invocation_type!" };
        case sycl::kernel_invocation_type::basic:
            // basic data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(::sycl::range<2>{ num_rows, num_rhs }, sycl::detail::basic::device_kernel_symm{ num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
            break;
        case sycl::kernel_invocation_type::work_group:
            // work-group data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::work_group::device_kernel_symm{ cgh, num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
            break;
        case sycl::kernel_invocation_type::hierarchical:
            // hierarchical data parallel kernels
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical::device_kernel_symm{ num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
            break;
        case sycl::kernel_invocation_type::scoped:
            throw exception{ "Can't use the sycl::kernel_invocation_type::scoped together with DPC++!" };
    }
#endif
    detail::device_synchronize(devices_[0]);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const unsigned long long num_rows_reduced = A_d.shape().x - 1;
    const unsigned long long num_features = A_d.shape().y;
    const unsigned long long num_classes = B_d.shape().x;

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::device_kernel_assembly_linear_symm{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes });
            });
            break;
        case kernel_function_type::polynomial:
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::device_kernel_assembly_polynomial_symm{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, num_features, QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value(), B_d.get(), C_d.get(), num_classes });
            });
            break;
        case kernel_function_type::rbf:
            devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::device_kernel_assembly_rbf_symm{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, num_features, QA_cost, cost_factor, params.gamma.value(), B_d.get(), C_d.get(), num_classes });
            });
            break;
    }
    detail::device_synchronize(devices_[0]);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = sv_d.shape().x;
    const unsigned long long num_features = sv_d.shape().y;

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE_OLD.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_features) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[0] };

    devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for(execution_range, sycl::detail::device_kernel_w_linear{ cgh, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv });
    });
    detail::device_synchronize(devices_[0]);

    return w_d;
}

auto csvm::run_predict_kernel(const parameter &params, const device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = sv_d.shape().x;
    const unsigned long long num_predict_points = predict_points_d.shape().x;
    const unsigned long long num_features = predict_points_d.shape().y;

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[0] };

    // define block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE_OLD.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };

    if (params.kernel_type == kernel_function_type::linear) {
        const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                     static_cast<std::size_t>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
        const ::sycl::nd_range<2> execution_range{ grid, block };

        devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            cgh.parallel_for(execution_range, sycl::detail::device_kernel_predict_linear{ cgh, out_d.get(), w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features });
        });
    } else {
        const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_sv) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                     static_cast<std::size_t>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
        const ::sycl::nd_range<2> execution_range{ grid, block };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    cgh.parallel_for(execution_range, sycl::detail::device_kernel_predict_polynomial{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.degree.value(), params.gamma.value(), params.coef0.value() });
                });
                break;
            case kernel_function_type::rbf:
                devices_[0].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    cgh.parallel_for(execution_range, sycl::detail::device_kernel_predict_rbf{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.gamma.value() });
                });
                break;
        }
    }
    detail::device_synchronize(devices_[0]);

    return out_d;
}

}  // namespace plssvm::dpcpp

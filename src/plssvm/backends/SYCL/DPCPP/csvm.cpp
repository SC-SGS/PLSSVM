/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"

#include "plssvm/backends/SYCL/DPCPP/detail/device_ptr.hpp"  // plssvm::dpcpp::detail::::device_ptr
#include "plssvm/backends/SYCL/DPCPP/detail/queue_impl.hpp"  // plssvm::dpcpp::detail::queue (PImpl implementation)
#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"     // plssvm::dpcpp::detail::get_device_list, plssvm::dpcpp::device_synchronize

#include "plssvm/backends/SYCL/exceptions.hpp"               // plssvm::dpcpp::backend_exception
#include "plssvm/backends/SYCL/predict_kernel.hpp"           // plssvm::sycl::detail::{kernel_w, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/backends/SYCL/q_kernel.hpp"                 // plssvm::sycl::detail::{device_kernel_q_linear, device_kernel_q_polynomial, device_kernel_q_rbf}
#include "plssvm/backends/SYCL/svm_kernel_hierarchical.hpp"  // plssvm::sycl::detail::{hierarchical_device_kernel_linear, hierarchical_device_kernel_polynomial, hierarchical_device_kernel_rbf}
#include "plssvm/backends/SYCL/svm_kernel_nd_range.hpp"      // plssvm::sycl::detail::{nd_range_device_kernel_linear, nd_range_device_kernel_polynomial, nd_range_device_kernel_rbf}
#include "plssvm/backends/gpu_csvm.hpp"                      // plssvm::detail::gpu_csvm
#include "plssvm/constants.hpp"                              // plssvm::kernel_index_type
#include "plssvm/detail/assert.hpp"                          // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"                 // plssvm::detail::execution_range
#include "plssvm/detail/logger.hpp"                          // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"             // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                  // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                  // plssvm::kernel_type
#include "plssvm/parameter.hpp"                              // plssvm::parameter
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "sycl/sycl.hpp"  // sycl::queue, sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <tuple>      // std::tie
#include <vector>     // std::vector

namespace plssvm::dpcpp {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

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
    target_platform used_target;
    std::tie(devices_, used_target) = detail::get_device_list(target);

    // set correct kernel invocation type if "automatic" has been provided
    if (invocation_type_ == sycl::kernel_invocation_type::automatic) {
        // always use nd_range for DPC++
        invocation_type_ = sycl::kernel_invocation_type::nd_range;
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing DPC++ ({}) as SYCL backend with the kernel invocation type \"{}\" for the svm_kernel.\n",
                        plssvm::detail::tracking_entry{ "backend", "version", __SYCL_COMPILER_VERSION },
                        plssvm::detail::tracking_entry{ "backend", "sycl_kernel_invocation_type", invocation_type_ });
    if (target == target_platform::automatic) {
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n", used_target);
    }
    PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::sycl }));
    PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "sycl_implementation_type", plssvm::sycl::implementation_type::dpcpp }));

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", used_target) };
    }

    // print found SYCL devices
    plssvm::detail::log(verbosity_level::full,
                        "\nFound {} SYCL device(s) for the target platform {}:\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking_entry{ "backend", "target_platform", used_target });
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}]\n", device, devices_[device].impl->sycl_queue.get_device().template get_info<::sycl::info::device::name>());
    }
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (queue_type &q : devices_) {
            device_synchronize(q);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what() << std::endl;
        std::terminate();
    }
}

void csvm::device_synchronize(const queue_type &queue) const {
    detail::device_synchronize(queue);
}

template <std::size_t I>
::sycl::nd_range<I> execution_range_to_native(const ::plssvm::detail::execution_range &range, const sycl::kernel_invocation_type invocation_type) {
    PLSSVM_ASSERT(invocation_type != sycl::kernel_invocation_type::automatic, "The SYCL kernel invocation type may not be automatic anymore at this point!");

    // set grid value based on used kernel invocation type
    const auto fill_grid = [&](const std::size_t i) {
        switch (invocation_type) {
            case sycl::kernel_invocation_type::nd_range:
                return range.grid[i] * range.block[i];
            case sycl::kernel_invocation_type::hierarchical:
                return range.grid[i];
            case sycl::kernel_invocation_type::automatic:
                throw backend_exception{ "Can't create native execution range from kernel invocation type automatic!" };
            default:
                throw backend_exception{ "Illegal kernel invocation type!" };
        }
    };

    if constexpr (I == 1) {
        ::sycl::range<1> grid{ fill_grid(0) };
        ::sycl::range<1> block{ range.block[0] };
        return ::sycl::nd_range<1>{ grid, block };
    } else if constexpr (I == 2) {
        ::sycl::range<2> grid{ fill_grid(0), fill_grid(1) };
        ::sycl::range<2> block{ range.block[0], range.block[1] };
        return ::sycl::nd_range<2>{ grid, block };
    } else if constexpr (I == 3) {
        ::sycl::range<3> grid{ fill_grid(0), fill_grid(1), fill_grid(2) };
        ::sycl::range<3> block{ range.block[0], range.block[1], range.block[2] };
        return ::sycl::nd_range<3>{ grid, block };
    } else {
        static_assert(I <= 3, "Illegal nd_range size!");
    }
}

template <typename real_type>
void csvm::run_q_kernel_impl(const std::size_t device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &q_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_data_points_padded, const std::size_t num_features) const {
    constexpr std::size_t boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            devices_[device].impl->sycl_queue.parallel_for(::sycl::range<1>{ num_data_points_padded - boundary_size }, sycl::detail::device_kernel_q_linear(q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features)));
            break;
        case kernel_function_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            devices_[device].impl->sycl_queue.parallel_for(::sycl::range<1>{ num_data_points_padded - boundary_size }, sycl::detail::device_kernel_q_polynomial(q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value()));
            break;
        case kernel_function_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            devices_[device].impl->sycl_queue.parallel_for(::sycl::range<1>{ num_data_points_padded - boundary_size }, sycl::detail::device_kernel_q_rbf(q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), params.gamma.value()));
            break;
    }
}

template void csvm::run_q_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t) const;
template void csvm::run_q_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_svm_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, const real_type QA_cost, const real_type add, const std::size_t num_data_points_padded, const std::size_t num_features) const {
    const ::sycl::nd_range execution_range = execution_range_to_native<2>(range, invocation_type_);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            devices_[device].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                if (invocation_type_ == sycl::kernel_invocation_type::nd_range) {
                    cgh.parallel_for(execution_range, sycl::detail::nd_range_device_kernel_linear(cgh, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, static_cast<kernel_index_type>(device)));
                } else if (invocation_type_ == sycl::kernel_invocation_type::hierarchical) {
                    cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical_device_kernel_linear(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, static_cast<kernel_index_type>(device)));
                }
            });
            break;
        case kernel_function_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            devices_[device].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                if (invocation_type_ == sycl::kernel_invocation_type::nd_range) {
                    cgh.parallel_for(execution_range, sycl::detail::nd_range_device_kernel_polynomial(cgh, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.degree.value(), params.gamma.value(), params.coef0.value()));
                } else if (invocation_type_ == sycl::kernel_invocation_type::hierarchical) {
                    cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical_device_kernel_polynomial(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.degree.value(), params.gamma.value(), params.coef0.value()));
                }
            });
            break;
        case kernel_function_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            devices_[device].impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                if (invocation_type_ == sycl::kernel_invocation_type::nd_range) {
                    cgh.parallel_for(execution_range, sycl::detail::nd_range_device_kernel_rbf(cgh, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.gamma.value()));
                } else if (invocation_type_ == sycl::kernel_invocation_type::hierarchical) {
                    cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), sycl::detail::hierarchical_device_kernel_rbf(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.gamma.value()));
                }
            });
            break;
    }
}

template void csvm::run_svm_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, const device_ptr_type<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, float, float, std::size_t, std::size_t) const;
template void csvm::run_svm_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, const device_ptr_type<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, double, double, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_w_kernel_impl(const std::size_t device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_data_points, const std::size_t num_features) const {
    devices_[device].impl->sycl_queue.parallel_for(::sycl::range<1>{ num_features }, sycl::detail::device_kernel_w_linear(w_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_data_points), static_cast<kernel_index_type>(num_features)));
}

template void csvm::run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t) const;
template void csvm::run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &out_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &point_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_support_vectors, const std::size_t num_predict_points, const std::size_t num_features) const {
    const ::sycl::nd_range execution_range = execution_range_to_native<2>(range, sycl::kernel_invocation_type::nd_range);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial:
            devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_predict_polynomial(out_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_support_vectors), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value()));

            break;
        case kernel_function_type::rbf:
            devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_predict_rbf(out_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_support_vectors), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.gamma.value()));
            break;
    }
}

template void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t, std::size_t) const;
template void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t, std::size_t) const;

}  // namespace plssvm::dpcpp

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

#include "plssvm/backends/SYCL/cg_explicit/blas.hpp"                     // plssvm::sycl::device_kernel_gemm
#include "plssvm/backends/SYCL/cg_explicit/kernel_matrix_assembly.hpp"   // plssvm::sycl::{device_kernel_assembly_linear, device_kernel_assembly_polynomial, device_kernel_assembly_rbf}
#include "plssvm/backends/SYCL/exceptions.hpp"                           // plssvm::dpcpp::backend_exception
#include "plssvm/backends/SYCL/predict_kernel.hpp"                       // plssvm::sycl::detail::{kernel_w, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/constants.hpp"                                          // plssvm::real_type
#include "plssvm/detail/assert.hpp"                                      // PLSSVM_ASSERT
#include "plssvm/detail/logger.hpp"                                      // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"                         // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                              // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                              // plssvm::kernel_type
#include "plssvm/parameter.hpp"                                          // plssvm::parameter
#include "plssvm/target_platforms.hpp"                                   // plssvm::target_platform

#include "fmt/core.h"                                        // fmt::format
#include "fmt/ostream.h"                                     // can use fmt using operator<< overloads
#include "sycl/sycl.hpp"                                     // sycl::queue, sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include <cstddef>                                           // std::size_t
#include <exception>                                         // std::terminate
#include <iostream>                                          // std::cout, std::endl
#include <tuple>                                             // std::tie
#include <vector>                                            // std::vector

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
    std::tie(devices_, target_) = detail::get_device_list(target);

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
                            "Using {} as automatic target platform.\n", target_);
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
                            "  [{}, {}]\n", device, device_name);
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

void csvm::device_synchronize(const queue_type &queue) const {
    detail::device_synchronize(queue);
}

unsigned long long csvm::get_device_memory() const {
    return devices_[0].impl->sycl_queue.get_device().get_info<::sycl::info::device::global_mem_size>();
}

[[nodiscard]] std::size_t csvm::get_max_work_group_size() const {
    return devices_[0].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_work_group_size>();
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const unsigned long long num_rows_reduced = data_d.size(0) - 1;
    const unsigned long long num_features = data_d.size(1);

    // define grid and block sizes
//    const std::size_t max_work_group_size = this->get_max_work_group_size();
//    const auto max_work_group_size_2D = static_cast<std::size_t>(std::sqrt(static_cast<real_type>(max_work_group_size)));
//    const ::sycl::range<2> block{ max_work_group_size_2D, max_work_group_size_2D };
//    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block[0]))) * block[0],
//                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block[1]))) * block[1] };
//    const ::sycl::nd_range<2> execution_range{ grid, block };
    const ::sycl::range<2> execution_range{ num_rows_reduced, num_rows_reduced };

    device_ptr_type kernel_matrix_d{ { num_rows_reduced, num_rows_reduced }, devices_[0] };
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_assembly_linear{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor });
            break;
        case kernel_function_type::polynomial:
            devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_assembly_polynomial{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, static_cast<real_type>(params.degree.value()), params.gamma.value(), params.coef0.value() });
            break;
        case kernel_function_type::rbf:
            devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_assembly_rbf{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value() });
            break;
    }
    devices_[0].impl->sycl_queue.wait_and_throw();

    return kernel_matrix_d;
}

void csvm::run_gemm_kernel_explicit(const std::size_t m, const std::size_t n, const std::size_t k, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    // define grid and block sizes
//    const std::size_t max_work_group_size = this->get_max_work_group_size();
//    const auto max_work_group_size_2D = static_cast<std::size_t>(std::sqrt(static_cast<real_type>(max_work_group_size)));
//    const ::sycl::range<2> block{ max_work_group_size_2D, max_work_group_size_2D };
//    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(m) / static_cast<double>(block[0]))) * block[0],
//                                 static_cast<std::size_t>(std::ceil(static_cast<double>(n) / static_cast<double>(block[1]))) * block[1] };
//    const ::sycl::nd_range<2> execution_range{ grid, block };
    const ::sycl::range<2> execution_range{ m, n };

    // cast to correct type
    const auto m_ull = static_cast<unsigned long long>(m);
    const auto n_ull = static_cast<unsigned long long>(n);
    const auto k_ull = static_cast<unsigned long long>(k);

    devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_gemm{ m_ull, n_ull, k_ull, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
    devices_[0].impl->sycl_queue.wait_and_throw();
}


//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.size(0);
    const unsigned long long num_sv = sv_d.size(0);
    const unsigned long long num_features = sv_d.size(1);

    const ::sycl::range<2> execution_range{ num_features, num_classes };

    device_ptr_type w_d{ { num_classes, num_features }, devices_[0] };

    devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_w_linear{ w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, num_features });
    devices_[0].impl->sycl_queue.wait_and_throw();

    return w_d;
}

auto csvm::run_predict_kernel(const parameter &params, const device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.size(0);
    const unsigned long long num_sv = sv_d.size(0);
    const unsigned long long num_predict_points = predict_points_d.size(0);
    const unsigned long long num_features = predict_points_d.size(1);

    device_ptr_type out_d{ { num_predict_points, num_classes }, devices_[0] };

    if (params.kernel_type == kernel_function_type::linear) {
        const ::sycl::range<2> execution_range{ num_predict_points, num_classes };

        devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_predict_linear{ out_d.get(), w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features });
    } else {
        const ::sycl::range<3> execution_range{ num_sv, num_predict_points, num_classes };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_predict_polynomial{ out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, static_cast<real_type>(params.degree.value()), params.gamma.value(), params.coef0.value() });
                break;
            case kernel_function_type::rbf:
                devices_[0].impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_predict_rbf{ out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.gamma.value() });
                break;
        }
    }
    devices_[0].impl->sycl_queue.wait_and_throw();

    return out_d;
}

}  // namespace plssvm::dpcpp

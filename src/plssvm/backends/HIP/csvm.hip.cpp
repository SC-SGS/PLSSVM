/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HIP/csvm.hpp"

#include "plssvm/backend_types.hpp"                                        // plssvm::backend_type
#include "plssvm/backends/HIP/cg_explicit/blas.hip.hpp"                    // plssvm::hip::device_kernel_gemm
#include "plssvm/backends/HIP/cg_explicit/kernel_matrix_assembly.hip.hpp"  // plssvm::hip::{device_kernel_assembly_linear, device_kernel_assembly_polynomial, device_kernel_assembly_rbf}
#include "plssvm/backends/HIP/cg_implicit/svm_kernel.hip.hpp"              // plssvm::hip::detail::{device_kernel_linear, device_kernel_polynomial, device_kernel_rbf}
#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"                   // plssvm::hip::detail::device_ptr
#include "plssvm/backends/HIP/detail/utility.hip.hpp"                      // plssvm::hip::detail::{device_synchronize, get_device_count, set_device, peek_at_last_error}
#include "plssvm/backends/HIP/exceptions.hpp"                              // plssvm::hip::backend_exception
#include "plssvm/backends/HIP/predict_kernel.hip.hpp"                      // plssvm::hip::detail::{device_kernel_w_linear, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/constants.hpp"                                            // plssvm::real_type
#include "plssvm/detail/assert.hpp"                                        // PLSSVM_ASSERT
#include "plssvm/detail/logger.hpp"                                        // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"                           // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                                // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                                // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                                            // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                                     // plssvm::target_platform

#include "fmt/core.h"                                     // fmt::format
#include "fmt/ostream.h"                                  // can use fmt using operator<< overloads
#include "hip/hip_runtime_api.h"                          // HIP runtime functions

#include <cstddef>                                        // std::size_t
#include <exception>                                      // std::terminate
#include <iostream>                                       // std::cout, std::endl
#include <numeric>                                        // std::iota
#include <utility>                                        // std::pair, std::make_pair

namespace plssvm::hip {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::gpu_amd) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the HIP backend!", target) };
    } else {
#if !defined(PLSSVM_HAS_AMD_TARGET)
        throw backend_exception{ fmt::format("Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing HIP as backend.\n");
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::hip }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "target_platform", plssvm::target_platform::gpu_amd }));

    // update the target platform
    target_ = plssvm::target_platform::gpu_amd;

    // get all available devices wrt the requested target platform
    devices_.resize(detail::get_device_count());
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no HIP devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "HIP backend selected but no HIP capable devices were found!" };
    }

    // print found HIP devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} HIP device(s):\n", plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() });
    for (const queue_type &device : devices_) {
        hipDeviceProp_t prop{};
        PLSSVM_HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, device));
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}, {}.{}]\n", device, prop.name, prop.major, prop.minor);
    }
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the HIP devices have finished before destruction
        for (const queue_type &device : devices_) {
            detail::device_synchronize(device);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

void csvm::device_synchronize(const queue_type &queue) const {
    detail::device_synchronize(queue);
}

unsigned long long csvm::get_device_memory() const {
    hipDeviceProp_t prop{};
    PLSSVM_HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, devices_[0]));
    return static_cast<unsigned long long>(prop.totalGlobalMem);
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const unsigned long long num_rows_reduced = data_d.size(0) - 1;
    const unsigned long long num_features = data_d.size(1);

    // define grid and block sizes
    const dim3 block(32, 32);
    const dim3 grid(static_cast<int>(std::ceil(num_rows_reduced / static_cast<double>(block.x))),
                    static_cast<int>(std::ceil(num_rows_reduced / static_cast<double>(block.y))));

    device_ptr_type kernel_matrix_d{ { num_rows_reduced, num_rows_reduced } };
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    detail::set_device(0);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            hip::device_kernel_assembly_linear<<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor);
            break;
        case kernel_function_type::polynomial:
            hip::device_kernel_assembly_polynomial<<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            hip::device_kernel_assembly_rbf<<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
    this->device_synchronize(devices_[0]);

    return kernel_matrix_d;
}

void csvm::run_gemm_kernel_explicit(const std::size_t m, const std::size_t n, const std::size_t k, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const dim3 block(32, 32);
    const dim3 grid(static_cast<int>(std::ceil(m / static_cast<double>(block.x))),
                    static_cast<int>(std::ceil(n / static_cast<double>(block.y))));

    detail::set_device(0);
    hip::device_kernel_gemm<<<grid, block>>>(m, n, k, alpha, A_d.get(), B_d.get(), beta, C_d.get());
    detail::peek_at_last_error();
    this->device_synchronize(devices_[0]);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.size(0);
    const std::size_t num_sv = sv_d.size(0);
    const std::size_t num_features = sv_d.size(1);

    const dim3 block(256, 4);
    const dim3 grid(static_cast<int>(std::ceil(num_features / static_cast<double>(block.x))),
                    static_cast<int>(std::ceil(num_classes / static_cast<double>(block.y))));

    device_ptr_type w_d{ { num_classes, num_features } };

    detail::set_device(0);
    hip::device_kernel_w_linear<<<grid, block>>>(w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, num_features);
    detail::peek_at_last_error();
    this->device_synchronize(devices_[0]);

    return w_d;
}

auto csvm::run_predict_kernel(const parameter &params, const device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.size(0);
    const std::size_t num_sv = sv_d.size(0);
    const std::size_t num_predict_points = predict_points_d.size(0);
    const std::size_t num_features = predict_points_d.size(1);

    device_ptr_type out_d{ { num_predict_points, num_classes } };

    detail::set_device(0);
    if (params.kernel_type == kernel_function_type::linear) {
        const dim3 block(256, 4);
        const dim3 grid(static_cast<int>(std::ceil(num_predict_points / static_cast<double>(block.x))),
                        static_cast<int>(std::ceil(num_classes / static_cast<double>(block.y))));

        hip::device_kernel_predict_linear<<<grid, block>>>(out_d.get(), w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features);
    } else {
        const dim3 block(16, 16, 4);
        const dim3 grid(static_cast<int>(std::ceil(num_sv / static_cast<double>(block.x))),
                        static_cast<int>(std::ceil(num_predict_points / static_cast<double>(block.y))),
                        static_cast<int>(std::ceil(num_classes / static_cast<double>(block.z))));

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                hip::device_kernel_predict_polynomial<<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), static_cast<kernel_index_type>(num_classes), static_cast<kernel_index_type>(num_sv), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value());
                break;
            case kernel_function_type::rbf:
                hip::device_kernel_predict_rbf<<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), static_cast<kernel_index_type>(num_classes), static_cast<kernel_index_type>(num_sv), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.gamma.value());
                break;
        }
    }
    detail::peek_at_last_error();
    this->device_synchronize(devices_[0]);

    return out_d;
}

}  // namespace plssvm::hip

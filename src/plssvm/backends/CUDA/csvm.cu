/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/csvm.hpp"

#include "plssvm/backend_types.hpp"                    // plssvm::backend_type
#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/CUDA/detail/utility.cuh"     // plssvm::cuda::detail::{device_synchronize, get_device_count, set_device, peek_at_last_error}
#include "plssvm/backends/CUDA/exceptions.hpp"         // plssvm::cuda::backend_exception
#include "plssvm/backends/CUDA/predict_kernel.cuh"     // plssvm::cuda::detail::{device_kernel_w_linear, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/backends/CUDA/q_kernel.cuh"           // plssvm::cuda::detail::{device_kernel_q_linear, device_kernel_q_polynomial, device_kernel_q_rbf}
#include "plssvm/backends/CUDA/svm_kernel.cuh"         // plssvm::cuda::detail::{device_kernel_linear, device_kernel_polynomial, device_kernel_rbf}
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"           // plssvm::detail::execution_range
#include "plssvm/detail/logger.hpp"                    // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"       // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/kernel_function_types.hpp"            // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "plssvm/backends/CUDA/blas.cuh"
#include "plssvm/backends/CUDA/kernel_matrix_assembly.cuh"

#include "cuda.h"                                      // cuda runtime functions
#include "cuda_runtime_api.h"                          // cuda runtime functions

#include "fmt/core.h"                                  // fmt::format
#include "fmt/ostream.h"                               // can use fmt using operator<< overloads

#include <cstddef>                                     // std::size_t
#include <exception>                                   // std::terminate
#include <iostream>                                    // std::cout, std::endl
#include <numeric>                                     // std::iota
#include <utility>                                     // std::pair, std::make_pair
#include <any>

namespace plssvm::cuda {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::gpu_nvidia) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the CUDA backend!", target) };
    } else {
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
        throw backend_exception{ "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!" };
#endif
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing CUDA as backend.\n");
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::cuda }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "target_platform", plssvm::target_platform::gpu_nvidia }));

    // update the target platform
    target_ = plssvm::target_platform::gpu_nvidia;

    // get all available devices wrt the requested target platform
    devices_.resize(detail::get_device_count());
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no CUDA devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "CUDA backend selected but no CUDA capable devices were found!" };
    }

    // print found CUDA devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} CUDA device(s):\n", plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() });
    for (const queue_type &device : devices_) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}, {}.{}]\n\n", device, prop.name, prop.major, prop.minor);
    }
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the CUDA devices have finished before destruction
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

std::pair<dim3, dim3> execution_range_to_native(const ::plssvm::detail::execution_range &range) {
    const dim3 grid(range.grid[0], range.grid[1], range.grid[2]);
    const dim3 block(range.block[0], range.block[1], range.block[2]);
    return std::make_pair(grid, block);
}


template <typename real_type>
auto csvm::run_w_kernel_impl(const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const -> device_ptr_type<real_type> {
    device_ptr_type<real_type> w_d{ num_classes * num_features };

    const dim3 block(256, 4);
    const dim3 grid(static_cast<int>(std::ceil(num_features / static_cast<double>(block.x))),
                    static_cast<int>(std::ceil(num_classes / static_cast<double>(block.y))));

    detail::set_device(0);
    cuda::device_kernel_w_linear<<<grid, block>>>(w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, num_features);
    detail::peek_at_last_error();

    return w_d;
}

template auto csvm::run_w_kernel_impl(const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t, std::size_t) const -> device_ptr_type<float>;
template auto csvm::run_w_kernel_impl(const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t, std::size_t) const -> device_ptr_type<double>;


template <typename real_type>
auto csvm::run_predict_kernel_impl(const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &rho_d, const device_ptr_type<real_type> &sv_d, const device_ptr_type<real_type> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const -> device_ptr_type<real_type> {
    device_ptr_type<real_type> out_d{ num_predict_points * num_classes };

    detail::set_device(0);
    if (params.kernel_type == kernel_function_type::linear) {
        const dim3 block(256, 4);
        const dim3 grid(static_cast<int>(std::ceil(num_predict_points / static_cast<double>(block.x))),
                        static_cast<int>(std::ceil(num_classes / static_cast<double>(block.y))));

        cuda::device_kernel_predict_linear<<<grid, block>>>(out_d.get(), w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features);
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
                cuda::device_kernel_predict_polynomial<<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), static_cast<kernel_index_type>(num_classes), static_cast<kernel_index_type>(num_sv), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value());
                break;
            case kernel_function_type::rbf:
                cuda::device_kernel_predict_rbf<<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), static_cast<kernel_index_type>(num_classes), static_cast<kernel_index_type>(num_sv), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.gamma.value());
                break;
        }
    }

    detail::peek_at_last_error();

    return out_d;
}

template auto csvm::run_predict_kernel_impl(const ::plssvm::detail::parameter<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t, std::size_t, std::size_t) const -> device_ptr_type<float>;
template auto csvm::run_predict_kernel_impl(const ::plssvm::detail::parameter<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t, std::size_t, std::size_t) const -> device_ptr_type<double>;



template <typename real_type>
::plssvm::detail::simple_any csvm::setup_data_on_devices_impl(const aos_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices may not be empty!");

    // initialize the data on the device
    device_ptr_type<real_type> data_d{ A.num_entries() };
    data_d.copy_to_device(A.data());

    return ::plssvm::detail::simple_any{ std::move(data_d) };
}

template ::plssvm::detail::simple_any csvm::setup_data_on_devices_impl(const aos_matrix<float> &) const;
template ::plssvm::detail::simple_any csvm::setup_data_on_devices_impl(const aos_matrix<double> &) const;

template <typename real_type>
::plssvm::detail::simple_any csvm::assemble_kernel_matrix_explicit_impl(const ::plssvm::detail::parameter<real_type> &params, const ::plssvm::detail::simple_any &data, const std::size_t num_rows_reduced, const std::size_t num_features, const std::vector<real_type> &q_red, real_type QA_cost) const {
    PLSSVM_ASSERT(num_rows_reduced > 0, "At least one row must be given!");
    PLSSVM_ASSERT(num_features > 0, "At least one feature must be given!");
    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector may not be empty!");

    // define grid and block sizes
    const dim3 block(32, 32);
    const dim3 grid(static_cast<int>(std::ceil(num_rows_reduced / static_cast<double>(block.x))),
                    static_cast<int>(std::ceil(num_rows_reduced / static_cast<double>(block.y))));

    // get the pointer to the data that already is on the device
    const device_ptr_type<real_type> &data_d = data.get<device_ptr_type<real_type>>();
    PLSSVM_ASSERT((num_rows_reduced + 1) * num_features == data_d.size(),
                  "The number of values on the device data array is {}, but the provided sizes are {} ((num_rows_reduced + 1) * num_features)",
                  data_d.size(), (num_rows_reduced + 1) * num_features);

    // allocate memory for the values currently not on the device
    device_ptr_type<real_type> q_red_d{ q_red.size() };
    q_red_d.copy_to_device(q_red);
    device_ptr_type<real_type> kernel_matrix{ num_rows_reduced * num_rows_reduced };

    detail::set_device(0);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            cuda::device_kernel_assembly_linear<<<grid, block>>>(q_red_d.get(), kernel_matrix.get(), data_d.get(), QA_cost, real_type{ 1.0 } / params.cost, static_cast<kernel_index_type>(num_rows_reduced), static_cast<kernel_index_type>(num_features));
            break;
        case kernel_function_type::polynomial:
            cuda::device_kernel_assembly_polynomial<<<grid, block>>>(q_red_d.get(), kernel_matrix.get(), data_d.get(), QA_cost, real_type{ 1.0 } / params.cost, static_cast<kernel_index_type>(num_rows_reduced), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            cuda::device_kernel_assembly_rbf<<<grid, block>>>(q_red_d.get(), kernel_matrix.get(), data_d.get(), QA_cost, real_type{ 1.0 } / params.cost, static_cast<kernel_index_type>(num_rows_reduced), static_cast<kernel_index_type>(num_features), params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
    detail::device_synchronize(0);

    PLSSVM_ASSERT(num_rows_reduced * num_rows_reduced == kernel_matrix.size(),
                  "The kernel matrix must be a quadratic matrix with num_rows_reduced^2 ({}) entries, but is {}!",
                  num_rows_reduced * num_rows_reduced, kernel_matrix.size());

    return ::plssvm::detail::simple_any{ std::move(kernel_matrix) };
}

template ::plssvm::detail::simple_any csvm::assemble_kernel_matrix_explicit_impl(const ::plssvm::detail::parameter<float> &, const ::plssvm::detail::simple_any &, const std::size_t, const std::size_t, const std::vector<float> &, float) const;
template ::plssvm::detail::simple_any csvm::assemble_kernel_matrix_explicit_impl(const ::plssvm::detail::parameter<double> &, const ::plssvm::detail::simple_any &, const std::size_t, const std::size_t, const std::vector<double> &, double) const;

template <typename real_type>
void csvm::kernel_gemm_explicit_impl(const real_type alpha, const ::plssvm::detail::simple_any &A, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) const {
    const device_ptr_type<real_type> &A_d = A.get<device_ptr_type<real_type>>();

    PLSSVM_ASSERT(!A_d.empty(), "The A matrix may not be empty!");
    PLSSVM_ASSERT(!B.empty(), "The B matrix may not be empty!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix may not be empty!");

    const std::size_t num_rhs = B.num_rows();
    const std::size_t num_rows = B.num_cols();

    // allocate memory on the device
    device_ptr_type<real_type> B_d{ B.num_entries() };
    B_d.copy_to_device(B.data());
    device_ptr_type<real_type> C_d{ C.num_entries() };
    C_d.copy_to_device(C.data());

    const dim3 block(32, 32);
    const dim3 grid(static_cast<int>(std::ceil(num_rows / static_cast<double>(block.x))),
                    static_cast<int>(std::ceil(num_rhs / static_cast<double>(block.y))));

    detail::set_device(0);
    cuda::device_kernel_gemm<<<grid, block>>>(num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get());
    detail::peek_at_last_error();
    detail::device_synchronize(0);

    C_d.copy_to_host(C.data());
}

template void csvm::kernel_gemm_explicit_impl(const float, const ::plssvm::detail::simple_any &, const aos_matrix<float> &, const float, aos_matrix<float> &) const;
template void csvm::kernel_gemm_explicit_impl(const double, const ::plssvm::detail::simple_any &, const aos_matrix<double> &, const double, aos_matrix<double> &) const;

}  // namespace plssvm::cuda

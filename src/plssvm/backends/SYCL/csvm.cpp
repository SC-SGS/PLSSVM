/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/csvm.hpp"

#include "plssvm/backends/SYCL/detail/device_ptr.hpp"  // plssvm::detail::sycl::device_ptr, plssvm::detail::sycl::get_device_list, plssvm::detail::sycl::device_synchronize
#include "plssvm/backends/SYCL/exceptions.hpp"         // plssvm::sycl::backend_exception
#include "plssvm/backends/SYCL/predict_kernel.hpp"     // plssvm::sycl::kernel_w, plssvm::sycl::predict_points_poly, plssvm::sycl::predict_points_rbf
#include "plssvm/backends/SYCL/q_kernel.hpp"           // plssvm::sycl::device_kernel_q_linear, plssvm::sycl::device_kernel_q_poly, plssvm::sycl::device_kernel_q_radial
#include "plssvm/backends/SYCL/svm_kernel.hpp"         // plssvm::sycl::device_kernel_linear, plssvm::sycl::device_kernel_poly, plssvm::sycl::device_kernel_radial
#include "plssvm/backends/gpu_csvm.hpp"                // plssvm::detail::gpu_csvm
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platform.hpp"                  // plssvm::target_platform

#include "fmt/core.h"     // fmt::print, fmt::format
#include "sycl/sycl.hpp"  // sycl::queue, sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <vector>     // std::vector

namespace plssvm::sycl {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    base_type{ params } {
    // check whether the requested target platform has been enabled
    switch (target_) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
    }

    if (print_info_) {
        fmt::print("Using SYCL as backend.\n");
    }

    // TODO: check multi GPU
    // get all available devices wrt the requested target platform
    devices_ = detail::get_device_list(target_);
    devices_.resize(std::min(devices_.size(), num_features_));

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", target_) };
    }

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        devices_.resize(1);
    }

    // resize vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found SYCL devices
        fmt::print("Found {} SYCL device(s) for the target platform {}:\n", devices_.size(), target_);
        for (size_type device = 0; device < devices_.size(); ++device) {
            fmt::print("  [{}, {}]\n", device, devices_[device].get_device().template get_info<::sycl::info::device::name>());
        }
        fmt::print("\n");
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (::sycl::queue &q : devices_) {
            detail::device_synchronize(q);
        }
    } catch (const plssvm::exception &e) {
        fmt::print("SYCL exception thrown: {}\n", e.what());
        std::terminate();
    }
}

template <typename T>
void csvm<T>::device_synchronize(queue_type &queue) {
    detail::device_synchronize(queue);
}

template <std::size_t I, typename size_type>
::sycl::nd_range<I> execution_range_to_native(const ::plssvm::detail::execution_range<size_type> &range) {
    if constexpr (I == 1) {
        ::sycl::range<1> grid{ range.grid[0] * range.block[0] };
        ::sycl::range<1> block{ range.block[0] };
        return ::sycl::nd_range<1>{ grid, block };
    } else if constexpr (I == 2) {
        ::sycl::range<2> grid{ range.grid[0] * range.block[0], range.grid[1] * range.block[1] };
        ::sycl::range<2> block{ range.block[0], range.block[1] };
        return ::sycl::nd_range<2>{ grid, block };
    } else if constexpr (I == 3) {
        ::sycl::range<3> grid{ range.grid[0] * range.block[0], range.grid[1] * range.block[1], range.grid[2] * range.block[2] };
        ::sycl::range<3> block{ range.block[0], range.block[1], range.block[2] };
        return ::sycl::nd_range<3>{ grid, block };
    } else {
        static_assert(::plssvm::detail::always_false_v<size_type>, "Illegal nd_range size!");
    }
}

template <typename T>
void csvm<T>::run_q_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, device_ptr_type &q_d, const int first_feature, const int last_feature) {
    const ::sycl::nd_range execution_range = execution_range_to_native<1>(range);
    switch (kernel_) {
        case kernel_type::linear:
            devices_[device].parallel_for(execution_range, device_kernel_q_linear{ q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, first_feature, last_feature });
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            devices_[device].parallel_for(execution_range, device_kernel_q_poly{ q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, degree_, gamma_, coef0_ });
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            devices_[device].parallel_for(execution_range, device_kernel_q_radial{ q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, gamma_ });
            break;
    }
}

template <typename T>
void csvm<T>::run_svm_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const int first_feature, const int last_feature) {
    const ::sycl::nd_range execution_range = execution_range_to_native<2>(range);
    switch (kernel_) {
        case kernel_type::linear:
            devices_[device].submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, device_kernel_linear{ cgh, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, add, first_feature, last_feature });
            });
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            devices_[device].submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, device_kernel_poly{ cgh, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_ });
            });
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            devices_[device].submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, device_kernel_radial{ cgh, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_ });
            });
            break;
    }
}

template <typename T>
void csvm<T>::run_w_kernel(const ::plssvm::detail::execution_range<size_type> &range, const device_ptr_type &alpha_d) {
    const ::sycl::nd_range execution_range = execution_range_to_native<1>(range);
    devices_[0].parallel_for(execution_range, device_kernel_w_linear{ w_d_.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, num_features_ });
}

template <typename T>
void csvm<T>::run_predict_kernel(const ::plssvm::detail::execution_range<size_type> &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const size_type num_predict_points) {
    const ::sycl::nd_range execution_range = execution_range_to_native<2>(range);
    switch (kernel_) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            devices_[0].parallel_for(execution_range, device_kernel_predict_poly{ out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, degree_, gamma_, coef0_ });
            break;
        case kernel_type::rbf:
            devices_[0].parallel_for(execution_range, device_kernel_predict_radial{ out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, gamma_ });
            break;
    }
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::sycl

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/csvm.hpp"

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/device_ptr.hpp" // plssvm::detail::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::device_ptr
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/utility.hpp"    // plssvm::detail::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::get_device_list, plssvm::detail::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::device_synchronize
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/exceptions.hpp"        // plssvm::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::backend_exception
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/constants.hpp"  // PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL, forward declaration and namespace alias
#include "plssvm/backends/SYCL/predict_kernel.hpp"                                  // plssvm::sycl_generic::kernel_w, plssvm::sycl_generic::predict_points_poly, plssvm::sycl_generic::predict_points_rbf
#include "plssvm/backends/SYCL/q_kernel.hpp"                                        // plssvm::sycl_generic::device_kernel_q_linear, plssvm::sycl_generic::device_kernel_q_poly, plssvm::sycl_generic::device_kernel_q_radial
#include "plssvm/backends/SYCL/svm_kernel_hierarchical.hpp"                         // plssvm::sycl_generic::hierarchical_device_kernel_linear, plssvm::sysycl_genericcl::hierarchical_device_kernel_poly, plssvm::sycl_generic::hierarchical_device_kernel_radial
#include "plssvm/backends/SYCL/svm_kernel_nd_range.hpp"                             // plssvm::sycl_generic::nd_range_device_kernel_linear, plssvm::sycl_generic::nd_range_device_kernel_poly, plssvm::sycl_generic::nd_range_device_kernel_radial
#include "plssvm/backends/gpu_csvm.hpp"                                             // plssvm::detail::gpu_csvm
#include "plssvm/constants.hpp"                                                     // plssvm::kernel_index_type
#include "plssvm/detail/assert.hpp"                                                 // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"                                        // plssvm::detail::execution_range
#include "plssvm/exceptions/exceptions.hpp"                                         // plssvm::exception
#include "plssvm/kernel_types.hpp"                                                  // plssvm::kernel_type
#include "plssvm/parameter.hpp"                                                     // plssvm::parameter
#include "plssvm/target_platforms.hpp"                                              // plssvm::target_platform

#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "sycl/sycl.hpp"  // sycl::queue, sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <tuple>      // std::tie
#include <vector>     // std::vector

namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@ {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    base_type{ params }, invocation_type_{ params.sycl_kernel_invocation_type } {
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

    // get all available devices wrt the requested target platform
    target_platform used_target;
    std::tie(devices_, used_target) = detail::get_device_list(target_);
    devices_.resize(std::min(devices_.size(), num_features_));

    // set correct kernel invocation type if "automatic" has been provided
    if (invocation_type_ == kernel_invocation_type::automatic) {
        // always use nd_range except for hipSYCL on the CPU
        if (used_target == target_platform::cpu && PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL) {
            invocation_type_ = kernel_invocation_type::hierarchical;
        } else {
            invocation_type_ = kernel_invocation_type::nd_range;
        }
    }

    if (print_info_) {
#if PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
        const auto sycl_compiler_version = ::hipsycl::sycl::detail::version_string();
#elif PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_DPCPP
        const auto sycl_compiler_version =  __SYCL_COMPILER_VERSION;
#endif
        fmt::print("Using SYCL ({}, {}) as backend with the kernel invocation type \"{}\" for the svm_kernel.\n", PLSSVM_SYCL_BACKEND_COMPILER_NAME, sycl_compiler_version, invocation_type_);
        if (target_ == target_platform::automatic) {
            fmt::print("Using {} as automatic target platform.\n", used_target);
        }
    }

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", used_target) };
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
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            fmt::print("  [{}, {}]\n", device, devices_[device]->get_device().template get_info<detail::sycl::info::device::name>());
        }
        fmt::print("\n");
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (queue_type& q : devices_) {
            device_synchronize(q);
        }
    } catch (const plssvm::exception &e) {
        fmt::print("SYCL exception thrown: {}\n", e.what());
        std::terminate();
    }
}

template <typename T>
void csvm<T>::device_synchronize(queue_type &queue) {
    detail::device_synchronize(*queue);
}

template <std::size_t I>
detail::sycl::nd_range<I> execution_range_to_native(const ::plssvm::detail::execution_range &range, const kernel_invocation_type invocation_type) {
    PLSSVM_ASSERT(invocation_type != kernel_invocation_type::automatic, "The SYCL kernel invocation type may not be automatic anymore at this point!");

    // set grid value based on used kernel invocation type
    const auto fill_grid = [&](const std::size_t i) {
        switch (invocation_type) {
            case kernel_invocation_type::nd_range:
                return range.grid[i] * range.block[i];
            case kernel_invocation_type::hierarchical:
                return range.grid[i];
            case kernel_invocation_type::automatic:
                throw backend_exception{ "Can't create native execution range from kernel invocation type automatic!" };
            default:
                throw backend_exception{ "Illegal kernel invocation type!" };
        }
    };

    if constexpr (I == 1) {
        detail::sycl::range<1> grid{ fill_grid(0) };
        detail::sycl::range<1> block{ range.block[0] };
        return detail::sycl::nd_range<1>{ grid, block };
    } else if constexpr (I == 2) {
        detail::sycl::range<2> grid{ fill_grid(0), fill_grid(1) };
        detail::sycl::range<2> block{ range.block[0], range.block[1] };
        return detail::sycl::nd_range<2>{ grid, block };
    } else if constexpr (I == 3) {
        detail::sycl::range<3> grid{ fill_grid(0), fill_grid(1), fill_grid(2) };
        detail::sycl::range<3> block{ range.block[0], range.block[1], range.block[2] };
        return detail::sycl::nd_range<3>{ grid, block };
    } else {
        static_assert(I <= 3, "Illegal nd_range size!");
    }
}

template <typename T>
void csvm<T>::run_q_kernel(const std::size_t device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, device_ptr_type &q_d, const std::size_t num_features) {
    switch (kernel_) {
        case kernel_type::linear:
            devices_[device]->parallel_for(detail::sycl::range<1>{ dept_ }, device_kernel_q_linear(q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_features));
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            devices_[device]->parallel_for(detail::sycl::range<1>{ dept_ }, device_kernel_q_poly(q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, degree_, gamma_, coef0_));
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            devices_[device]->parallel_for(detail::sycl::range<1>{ dept_ }, device_kernel_q_radial(q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, gamma_));
            break;
    }
}

template <typename T>
void csvm<T>::run_svm_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const std::size_t num_features) {
    const detail::sycl::nd_range execution_range = execution_range_to_native<2>(range, invocation_type_);
    switch (kernel_) {
        case kernel_type::linear:
            devices_[device]->submit([&](detail::sycl::handler &cgh) {
                if (invocation_type_ == kernel_invocation_type::nd_range) {
                    cgh.parallel_for(execution_range, nd_range_device_kernel_linear(cgh, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_features, add, device));
                } else if (invocation_type_ == kernel_invocation_type::hierarchical) {
                    cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), hierarchical_device_kernel_linear(q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_features, add, device));
                }
            });
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            devices_[device]->submit([&](detail::sycl::handler &cgh) {
                if (invocation_type_ == kernel_invocation_type::nd_range) {
                    cgh.parallel_for(execution_range, nd_range_device_kernel_poly(cgh, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_));
                } else if (invocation_type_ == kernel_invocation_type::hierarchical) {
                    cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), hierarchical_device_kernel_poly(q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_));
                }
            });
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            devices_[device]->submit([&](detail::sycl::handler &cgh) {
                if (invocation_type_ == kernel_invocation_type::nd_range) {
                    cgh.parallel_for(execution_range, nd_range_device_kernel_radial(cgh, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_));
                } else if (invocation_type_ == kernel_invocation_type::hierarchical) {
                    cgh.parallel_for_work_group(execution_range.get_global_range(), execution_range.get_local_range(), hierarchical_device_kernel_radial(q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_));
                }
            });
            break;
    }
}

template <typename T>
void csvm<T>::run_w_kernel(const std::size_t device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const std::size_t num_features) {
    devices_[device]->parallel_for(detail::sycl::range<1>{ num_features_ }, device_kernel_w_linear(w_d.get(), data_d_[device].get(), data_last_d_[device].get(), alpha_d.get(), num_data_points_, num_features));
}

template <typename T>
void csvm<T>::run_predict_kernel(const ::plssvm::detail::execution_range &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const std::size_t num_predict_points) {
    const detail::sycl::nd_range execution_range = execution_range_to_native<2>(range, kernel_invocation_type::nd_range);
    switch (kernel_) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            devices_[0]->parallel_for(execution_range, device_kernel_predict_poly(out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, degree_, gamma_, coef0_));

            break;
        case kernel_type::rbf:
            devices_[0]->parallel_for(execution_range, device_kernel_predict_radial(out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, gamma_));
            break;
    }
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@

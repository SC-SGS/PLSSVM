/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Factory function for constructing a new C-SVM based on the provided `plssvm::backend_type` and potential additional parameter.
 */

#ifndef PLSSVM_CSVM_FACTORY_HPP_
#define PLSSVM_CSVM_FACTORY_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                      // plssvm::backend
#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl::implementation_type
#include "plssvm/csvm.hpp"                               // plssvm::csvm, plssvm::csvm_backend_exists_v
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::remove_cvref_t
#include "plssvm/exceptions/exceptions.hpp"              // plssvm::unsupported_backend_exception
#include "plssvm/kernel_function_types.hpp"              // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                          // plssvm::parameter
#include "plssvm/target_platforms.hpp"                   // plssvm::target_platform

// only include requested/available backends
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/csvm.hpp"  // plssvm::cuda::csvm
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    #include "plssvm/backends/HIP/csvm.hpp"  // plssvm::hip::csvm
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/csvm.hpp"  // plssvm::opencl::csvm
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
        #include "plssvm/backends/SYCL/DPCPP/csvm.hpp"  // plssvm::dpcpp::csvm
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
        #include "plssvm/backends/SYCL/hipSYCL/csvm.hpp"  // plssvm::hipsycl::csvm
    #endif
namespace plssvm::sycl {
using namespace plssvm::PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION;
}
#endif

#include "igor/igor.hpp"  // igor::parser, igor::has_unnamed_arguments

#include <memory>       // std::unique_ptr, std::make_unique
#include <type_traits>  // std::is_same_v, std::enable_if_t
#include <utility>      // std::forward

#include "plssvm/detail/utility.hpp"
#include <iostream>

namespace plssvm {

namespace detail {

/**
 * @brief Construct a C-SVM using the parameters @p args.
 * @details The default case, no special parameters for the C-SVMs are necessary.
 * @tparam csvm_type the type of the C-SVM
 * @tparam Args the types of the parameters to initialize the C-SVM
 * @param[in] args the parameters used to initialize the C-SVM
 * @return the C-SVM (`[[nodiscard]]`)
 */
template <typename csvm_type, typename... Args>
[[nodiscard]] std::unique_ptr<::plssvm::csvm> make_csvm_default_impl([[maybe_unused]] Args &&...args) {
    // test whether the backend is available
    if constexpr (csvm_backend_exists_v<csvm_type>) {
        // test whether the backend can be constructed with the provided parameter
        if constexpr (std::is_constructible_v<csvm_type, Args...>) {
            return std::make_unique<csvm_type>(std::forward<Args>(args)...);
        } else {
            throw unsupported_backend_exception{ fmt::format("Provided invalid (named) arguments for the {} backend!", plssvm::csvm_to_backend_type_v<csvm_type>) };
        }
    } else {
        throw unsupported_backend_exception{ fmt::format("No {} backend available!", plssvm::csvm_to_backend_type_v<csvm_type>) };
    }
}


/**
 * @brief Construct a SYCL C-SVM using the parameters @p args.
 * @details The special case for the SYCL backend to handle the SYCL specific parameters.
 * @tparam Args the types of the parameters to initialize the SYCL C-SVM
 * @param[in] args the parameters used to initialize the SYCL C-SVM
 * @return the SYCL C-SVM (`[[nodiscard]]`)
 */
template <typename... Args>
[[nodiscard]] std::unique_ptr<::plssvm::csvm> make_csvm_sycl_impl([[maybe_unused]] Args &&...args) {
    // check igor parameter
    igor::parser parser{ args... };

    // get the SYCL implementation type to use
    sycl::implementation_type impl_type = sycl::implementation_type::automatic;
    // check whether a specific SYCL implementation type has been requested
    if constexpr (parser.has(sycl_implementation_type)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_same_v<::plssvm::detail::remove_cvref_t<decltype(parser(sycl_implementation_type))>, sycl::implementation_type>, "Provided sycl_implementation_type must be convertible to a plssvm::sycl::implementation_type!");
        impl_type = static_cast<sycl::implementation_type>(parser(sycl_implementation_type));
    }
    // TODO: check incomplete type (after constructor fix)
    switch (impl_type) {
        case sycl::implementation_type::automatic:
            return make_csvm_default_impl<sycl::csvm>(std::forward<Args>(args)...);
        case sycl::implementation_type::dpcpp:
            return make_csvm_default_impl<dpcpp::csvm>(std::forward<Args>(args)...);
        case sycl::implementation_type::hipsycl:
            return make_csvm_default_impl<hipsycl::csvm>(std::forward<Args>(args)...);
    }
}

/**
 * @brief Create a new C-SVM using the @p backend type and the additional parameter @p args.
 * @tparam Args the types of the parameters to initialize the C-SVM
 * @param[in] backend the backend to use
 * @param[in] args the parameters used to initialize the respective C-SVM
 * @return the C-SVM (`[[nodiscard]]`)
 */
template <typename... Args>
[[nodiscard]] std::unique_ptr<::plssvm::csvm> make_csvm_impl(const backend_type backend, Args &&...args) {
    switch (backend) {
        case backend_type::automatic:
            return make_csvm_impl(determine_default_backend(), std::forward<Args>(args)...);
        case backend_type::openmp:
            return make_csvm_default_impl<openmp::csvm>(std::forward<Args>(args)...);
        case backend_type::cuda:
            return make_csvm_default_impl<cuda::csvm>(std::forward<Args>(args)...);
        case backend_type::hip:
            return make_csvm_default_impl<hip::csvm>(std::forward<Args>(args)...);
        case backend_type::opencl:
            return make_csvm_default_impl<opencl::csvm>(std::forward<Args>(args)...);
        case backend_type::sycl:
            return make_csvm_sycl_impl(std::forward<Args>(args)...);
    }
    throw unsupported_backend_exception{ "Can't recognize backend !" };
}

}  // namespace detail

/**
 * @brief Create a new C-SVM using the @p backend type, the automatic target platform, and additional parameter @p args.
 * @tparam Args the types of the parameters to initialize the C-SVM
 * @param[in] backend the backend to use
 * @param[in] args the parameters used to initialize the respective C-SVM
 * @return the C-SVM (`[[nodiscard]]`)
 */
template <typename... Args>
[[nodiscard]] std::unique_ptr<::plssvm::csvm> make_csvm(const backend_type backend, Args &&...args) {
    return ::plssvm::detail::make_csvm_impl(backend, std::forward<Args>(args)...);
}
/**
 * @brief Create a new C-SVM using the automatic backend type, the automatic target platform, and the additional parameter @p args.
 * @tparam Args the types of the parameters to initialize the C-SVM
 * @param[in] args the parameters used to initialize the respective C-SVM
 * @return the C-SVM (`[[nodiscard]]`)
 */
template <typename... Args>
[[nodiscard]] inline std::unique_ptr<::plssvm::csvm> make_csvm(Args &&...args) {
    return ::plssvm::detail::make_csvm_impl(backend_type::automatic, std::forward<Args>(args)...);
}

}  // namespace plssvm

#endif  // PLSSVM_CSVM_FACTORY_HPP_

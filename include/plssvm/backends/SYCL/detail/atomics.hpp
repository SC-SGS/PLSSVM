/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Defines an atomic_ref wrapper for the SYCL backend.
*/

#pragma once

#include "plssvm/backends/SYCL/detail/constants.hpp"  // PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL, PLSSVM_SYCL_BACKEND_COMPILER_DPCPP

#include "sycl/sycl.hpp"  // sycl::atomic_ref, sycl::memory_order, sycl::memory_scope, sycl::access::address_space

namespace plssvm::sycl {
namespace detail {

// TODO: remove #if after Intel has a SYCL2020 conformant sycl::atomic_ref implementation
#if PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_DPCPP
using ::sycl::ext::oneapi::atomic_ref;
#elif PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
using ::sycl::atomic_ref;
#endif

}  // namespace detail

/**
 * @brief Shortcut alias for a [`sycl::atomic_ref`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:atomic-references).
 * @tparam T the type of the accessed values
 */
template <typename T>
using atomic_op = detail::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>;

}  // namespace plssvm::sycl
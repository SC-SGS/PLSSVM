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

#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_ATOMICS_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_ATOMICS_HPP_
#pragma once

#include "sycl/sycl.hpp"  // sycl::atomic_ref, sycl::memory_order, sycl::memory_scope, sycl::access::address_space

namespace plssvm::sycl::detail {

/**
 * @brief Shortcut alias for a [`sycl::atomic_ref`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:atomic-references) targeting global memory.
 * @tparam T the type of the accessed values
 */
template <typename T>
using atomic_op = ::sycl::atomic_ref<T, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>;

}  // namespace plssvm::sycl

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_ATOMICS_HPP_
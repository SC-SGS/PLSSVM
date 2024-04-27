/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the stdpar backend.
 */

#ifndef PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_
#pragma once

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
    #include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::detail::atomic_op
#else
    // TODO: other stdpar implementations
    #include <atomic>  // std::atomic_ref
#endif

#include <string>  // std::string

namespace plssvm::stdpar::detail {

#if defined(PLSSVM_STDPAR_BACKEND_HAS_ACPP)
template <typename T>
using atomic_ref = plssvm::sycl::detail::atomic_op<T>;
#else
using std::atomic_ref;
#endif

/**
 * @brief Return the stdpar implementation used.
 * @return the stdpar implementation (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_stdpar_implementation();

}  // namespace plssvm::stdpar::detail

#endif  // PLSSVM_BACKENDS_STDPAR_DETAIL_UTILITY_HPP_

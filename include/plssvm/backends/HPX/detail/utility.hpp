/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the HPX backend.
 */

#ifndef PLSSVM_BACKENDS_HPX_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_HPX_DETAIL_UTILITY_HPP_
#pragma once

#include "boost/atomic/atomic_ref.hpp"  // boost::atomic_ref

#include <string>  // std::string

namespace plssvm::hpx::detail {

using boost::atomic_ref;

/**
 * @brief Return the number of used CPU threads in the HPX backend.
 * @return the number of used CPU threads (`[[nodiscard]]`)
 */
[[nodiscard]] int get_num_threads();

/**
 * @brief Return the HPX version used.
 * @return the HPX version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_hpx_version();

/**
 * @brief Start the runtime of the HPX backend.
 */
void start_hpx_runtime();

/**
 * @brief Stop the runtime of the HPX backend.
 */
void stop_hpx_runtime();

/**
 * @brief Scope Guard that leverages RAII to start and correctly teardown the HPX runtime even in case of exceptions.
 */
struct scope_guard
{
  scope_guard(){
    start_hpx_runtime();
  }
  ~scope_guard()
  {
    stop_hpx_runtime();
  }
};
}  // namespace plssvm::hpx::detail

#endif  // PLSSVM_BACKENDS_HPX_DETAIL_UTILITY_HPP_

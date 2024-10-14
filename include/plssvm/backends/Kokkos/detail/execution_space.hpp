/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Execution space enumeration for the ExecutionSpaces in Kokkos.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_EXECUTION_SPACE_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_EXECUTION_SPACE_HPP_
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>  // std::ostream forward declaration

namespace plssvm::kokkos::detail {

enum class execution_space {
    cuda,
    hip,
    sycl,
    hpx,
    openmp,
    openmp_target,
    openacc,
    threads,
    serial
};

std::ostream &operator<<(std::ostream &out, execution_space space);

}  // namespace plssvm::kokkos::detail

template <>
struct fmt::formatter<plssvm::kokkos::detail::execution_space> : fmt::ostream_formatter { };

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_EXECUTION_SPACE_HPP_

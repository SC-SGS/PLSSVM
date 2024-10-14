/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/execution_space.hpp"

#include <ostream>  // std::ostream

namespace plssvm::kokkos::detail {

std::ostream &operator<<(std::ostream &out, const execution_space space) {
    switch (space) {
        case execution_space::cuda:
            return out << "Cuda";
        case execution_space::hip:
            return out << "HIP";
        case execution_space::sycl:
            return out << "SYCL";
        case execution_space::hpx:
            return out << "HPX";
        case execution_space::openmp:
            return out << "OpenMP";
        case execution_space::openmp_target:
            return out << "OpenMPTarget";
        case execution_space::openacc:
            return out << "OpenACC";
        case execution_space::threads:
            return out << "Threads";
        case execution_space::serial:
            return out << "Serial";
    }
    return out << "unknown";
}

}  // namespace plssvm::kokkos::detail

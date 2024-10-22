/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/execution_space.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::contains

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::kokkos {

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

std::istream &operator>>(std::istream &in, execution_space &space) {
    std::string str{};
    in >> str;
    ::plssvm::detail::to_lower_case(str);

    if (str == "cuda") {
        space = execution_space::cuda;
    } else if (str == "hip") {
        space = execution_space::hip;
    } else if (str == "sycl") {
        space = execution_space::sycl;
    } else if (str == "hpx") {
        space = execution_space::hpx;
    } else if (str == "openmp") {
        space = execution_space::openmp;
    } else if (str == "openmp_target") {
        space = execution_space::openmp_target;
    } else if (str == "openacc") {
        space = execution_space::openacc;
    } else if (str == "threads") {
        space = execution_space::threads;
    } else if (str == "serial") {
        space = execution_space::serial;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::kokkos

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/execution_space.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::unreachable

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace, Kokkos macros, Kokkos ExecutionSpace types

#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

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
    } else if (str == "openmp_target" || str == "openmptarget") {
        space = execution_space::openmp_target;
    } else if (str == "openacc") {
        space = execution_space::openacc;
    } else if (str == "threads" || str == "std::threads") {
        space = execution_space::threads;
    } else if (str == "serial") {
        space = execution_space::serial;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

execution_space determine_execution_space() noexcept {
    // determine the execution_space enumeration value based on the provided Kokkos execution space
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>) {
        return execution_space::cuda;
    }
#endif
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>) {
        return execution_space::hip;
    }
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::SYCL>) {
        return execution_space::sycl;
    }
#endif
#if defined(KOKKOS_ENABLE_HPX)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::HPX>) {
        return execution_space::hpx;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMP>) {
        return execution_space::openmp;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::OpenMPTarget>) {
        return execution_space::openmp_target;
    }
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Experimental::OpenACC>) {
        return execution_space::openacc;
    }
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Threads>) {
        return execution_space::threads;
    }
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Serial>) {
        return execution_space::serial;
    }
#endif
    // at least one execution space must always be available!
    ::plssvm::detail::unreachable();
}

[[nodiscard]] std::vector<execution_space> available_execution_spaces() {
    std::vector<execution_space> available_spaces{};
#if defined(KOKKOS_ENABLE_CUDA)
    available_spaces.push_back(execution_space::cuda);
#endif
#if defined(KOKKOS_ENABLE_HIP)
    available_spaces.push_back(execution_space::hip);
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    available_spaces.push_back(execution_space::sycl);
#endif
#if defined(KOKKOS_ENABLE_HPX)
    available_spaces.push_back(execution_space::hpx);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    available_spaces.push_back(execution_space::openmp);
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    available_spaces.push_back(execution_space::openmp_target);
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
    available_spaces.push_back(execution_space::openacc);
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    available_spaces.push_back(execution_space::threads);
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    available_spaces.push_back(execution_space::serial);
#endif

    // AT LEAST ONE execution space must ALWAYS be available
    PLSSVM_ASSERT(!available_spaces.empty(), "Aat least one execution space must always be available!");

    return available_spaces;
}

}  // namespace plssvm::kokkos

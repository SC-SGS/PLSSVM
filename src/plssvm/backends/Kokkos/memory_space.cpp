/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/memory_space.hpp"

#include "plssvm/backends/Kokkos/detail/constexpr_available_memory_spaces.hpp"  // plssvm::kokkos::detail::constexpr_available_memory_spaces
#include "plssvm/detail/string_utility.hpp"                                     // plssvm::detail::to_lower_case

#include <array>    // std::array
#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::kokkos {

std::ostream &operator<<(std::ostream &out, const memory_space space) {
    switch (space) {
        case memory_space::host_space:
            return out << "HostSpace";
        case memory_space::cuda_space:
            return out << "CudaSpace";
        case memory_space::cuda_usm_space:
            return out << "CudaUVMSpace";
        case memory_space::hip_space:
            return out << "HIPSpace";
        case memory_space::hip_usm_space:
            return out << "HIPManagedSpace";
        case memory_space::sycl_space:
            return out << "SYCLDeviceUSMSpace";
        case memory_space::sycl_usm_space:
            return out << "SYCLSharedUSMSpace";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, memory_space &space) {
    std::string str{};
    in >> str;
    ::plssvm::detail::to_lower_case(str);

    if (str == "hostspace" || str == "host_space") {
        space = memory_space::host_space;
    } else if (str == "cudaspace" || str == "cuda_space") {
        space = memory_space::cuda_space;
    } else if (str == "cudauvmspace" || str == "cuda_uvm_space" || str == "cudausmspace" || str == "cuda_usm_space") {
        space = memory_space::cuda_usm_space;
    } else if (str == "hipspace" || str == "hip_space") {
        space = memory_space::hip_space;
    } else if (str == "hipmanagedspace" || str == "hip_managed_space" || str == "hipusmspace" || str == "hip_usm_space") {
        space = memory_space::hip_usm_space;
    } else if (str == "sycldeviceusmspace" || str == "sycl_device_usm_space" || str == "syclspace" || str == "sycl_space") {
        space = memory_space::sycl_space;
    } else if (str == "syclsharedusmspace" || str == "sycl_shared_usm_space" || str == "syclusmspace" || str == "sycl_usm_space") {
        space = memory_space::sycl_usm_space;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

std::vector<memory_space> list_available_memory_spaces() {
    // get all available memory spaces
    constexpr auto arr = detail::constexpr_available_memory_spaces();
    return std::vector<memory_space>{ arr.begin(), arr.end() };
}

}  // namespace plssvm::kokkos

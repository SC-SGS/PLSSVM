/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines all possible backends. Can also include backends not available on the target platform.
 */

#pragma once

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm {

/**
 * @brief Enum class for the different backend types.
 */
enum class backend_type {
    /** [OpenMP](https://www.openmp.org/) */
    openmp = 0,
    /** [CUDA](https://developer.nvidia.com/cuda-zone) */
    cuda = 1,
    /** [OpenCL](https://www.khronos.org/opencl/) */
    opencl = 2,
    /** [SYCL](https://www.khronos.org/sycl/) */
    sycl = 3
};

/**
 * @brief Stream-insertion-operator overload for convenient printing of the backend type @p backend.
 * @param[in,out] out the output-stream to write the backend type to
 * @param[in] backend the backend type
 * @return the output-stream
 */
inline std::ostream &operator<<(std::ostream &out, const backend_type backend) {
    switch (backend) {
        case backend_type::openmp:
            return out << "openmp";
        case backend_type::cuda:
            return out << "cuda";
        case backend_type::opencl:
            return out << "opencl";
        case backend_type::sycl:
            return out << "sycl";
    }
    return out << "unknown";
}

/**
 * @brief Stream-extraction-operator overload for convenient converting a string to a backend type.
 * @param[in,out] in input-stream to extract the backend type from
 * @param[in] backend the backend type
 * @return the input-stream
 */
inline std::istream &operator>>(std::istream &in, backend_type &backend) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "openmp") {
        backend = backend_type::openmp;
    } else if (str == "cuda") {
        backend = backend_type::cuda;
    } else if (str == "opencl") {
        backend = backend_type::opencl;
    } else if (str == "sycl") {
        backend = backend_type::sycl;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm

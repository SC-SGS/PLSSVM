/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all possible backends. Can also include backends not available on the target platform.
 */

#pragma once

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <ostream>      // std::ostream
#include <string_view>  // std::string_view

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
    opencl = 2
};

/**
 * @brief Output-operator overload for convenient printing of the backend type @p backend.
 * @param[inout] out the output-stream to write the backend type to
 * @param[in] backend the backend type
 * @return the output-stream
 */
inline std::ostream &operator<<(std::ostream &out, const backend_type backend) {
    switch (backend) {
        case backend_type::openmp:
            return out << "OpenMP";
        case backend_type::cuda:
            return out << "CUDA";
        case backend_type::opencl:
            return out << "OpenCL";
        default:
            return out << "unknown";
    }
}

}  // namespace plssvm

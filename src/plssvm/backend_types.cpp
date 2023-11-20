/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception

#include "fmt/format.h"  // fmt::format, fmt::join

#include <array>    // std::array
#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <utility>  // std::pair
#include <vector>   // std::vector

namespace plssvm {

std::vector<backend_type> list_available_backends() {
    std::vector<backend_type> available_backends = { backend_type::automatic };
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    available_backends.push_back(backend_type::openmp);
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    available_backends.push_back(backend_type::cuda);
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    available_backends.push_back(backend_type::hip);
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    available_backends.push_back(backend_type::opencl);
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    available_backends.push_back(backend_type::sycl);
#endif
    return available_backends;
}

backend_type determine_default_backend(const std::vector<backend_type> &available_backends, const std::vector<target_platform> &available_target_platforms) {
    // the decision order based on empiric findings
    using decision_order_type = std::pair<target_platform, std::vector<backend_type>>;
    const std::array decision_order = {
        decision_order_type{ target_platform::gpu_nvidia, { backend_type::cuda, backend_type::hip, backend_type::opencl, backend_type::sycl } },
        decision_order_type{ target_platform::gpu_amd, { backend_type::hip, backend_type::opencl, backend_type::sycl } },
        decision_order_type{ target_platform::gpu_intel, { backend_type::sycl, backend_type::opencl } },
        decision_order_type{ target_platform::cpu, { backend_type::sycl, backend_type::opencl, backend_type::openmp } }
    };

    // return the default backend based on the previously defined decision order
    for (const auto &[target, backends] : decision_order) {
        if (detail::contains(available_target_platforms, target)) {
            for (const backend_type b : backends) {
                if (detail::contains(available_backends, b)) {
                    return b;
                }
            }
        }
    }
    throw unsupported_backend_exception{ fmt::format("Error: unsupported backend and target platform combination: [{}]x[{}]!",
                                                     fmt::join(available_backends, ", "),
                                                     fmt::join(available_target_platforms, ", ")) };
}

std::ostream &operator<<(std::ostream &out, const backend_type backend) {
    switch (backend) {
        case backend_type::automatic:
            return out << "automatic";
        case backend_type::openmp:
            return out << "openmp";
        case backend_type::cuda:
            return out << "cuda";
        case backend_type::hip:
            return out << "hip";
        case backend_type::opencl:
            return out << "opencl";
        case backend_type::sycl:
            return out << "sycl";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, backend_type &backend) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        backend = backend_type::automatic;
    } else if (str == "openmp") {
        backend = backend_type::openmp;
    } else if (str == "cuda") {
        backend = backend_type::cuda;
    } else if (str == "hip") {
        backend = backend_type::hip;
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
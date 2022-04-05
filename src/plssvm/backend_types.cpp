/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"

#include "plssvm/target_platforms.hpp"  // plssvm::list_available_target_platforms

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception

#include <algorithm>  // std::find
#include <array>      // std::array
#include <ios>        // std::ios::failbit
#include <istream>    // std::istream
#include <ostream>    // std::ostream
#include <string>     // std::string
#include <vector>     // std::vector

namespace plssvm {

std::vector<backend_type> list_available_backends() {
    std::vector<backend_type> available_backends;
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

template <typename T>
bool contains(const std::vector<T> &vec, const T val) {
    return std::find(vec.begin(), vec.end(), val) != vec.end();
}

backend_type determine_default_backend() {
    std::vector<backend_type> available_backends = list_available_backends();
    std::vector<target_platform> available_target_platforms = list_available_target_platforms();

    using decision_order_type = std::pair<target_platform, std::vector<backend_type>>;

    std::array<decision_order_type, 4> decision_order = {
        decision_order_type{ target_platform::gpu_nvidia, { backend_type::cuda, backend_type::opencl, backend_type::sycl } },  // TODO: HIP?
        decision_order_type{ target_platform::gpu_amd, { backend_type::hip, backend_type::opencl, backend_type::sycl } },
        decision_order_type{ target_platform::gpu_intel, { backend_type::sycl, backend_type::opencl } },
        decision_order_type{ target_platform::cpu, { backend_type::sycl, backend_type::opencl, backend_type::openmp } }
    };

    for (const auto &[target, backends] : decision_order) {
        if (contains(available_target_platforms, target)) {
            for (const backend_type b : backends) {
                if (contains(available_backends, b)) {
                    return b;
                }
            }
        }
    }
    throw plssvm::unsupported_backend_exception{ "Unreachable!" };  // TODO: ?!
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

/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const backend_type backend) {
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

std::istream &operator>>(std::istream &in, backend_type &backend) {
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
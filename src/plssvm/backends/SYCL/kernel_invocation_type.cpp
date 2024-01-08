/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::sycl {

std::ostream &operator<<(std::ostream &out, const kernel_invocation_type target) {
    switch (target) {
        case kernel_invocation_type::automatic:
            return out << "automatic";
        case kernel_invocation_type::basic:
            return out << "basic";
        case kernel_invocation_type::work_group:
            return out << "work_group";
        case kernel_invocation_type::hierarchical:
            return out << "hierarchical";
        case kernel_invocation_type::scoped:
            return out << "scoped";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, kernel_invocation_type &target) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        target = kernel_invocation_type::automatic;
    } else if (str == "basic") {
        target = kernel_invocation_type::basic;
    } else if (str == "work_group") {
        target = kernel_invocation_type::work_group;
    } else if (str == "hierarchical") {
        target = kernel_invocation_type::hierarchical;
    } else if (str == "scoped") {
        target = kernel_invocation_type::scoped;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::sycl

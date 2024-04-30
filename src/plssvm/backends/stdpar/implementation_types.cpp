/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/implementation_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::stdpar {

std::ostream &operator<<(std::ostream &out, const implementation_type impl) {
    switch (impl) {
        case implementation_type::adaptivecpp:
            return out << "adaptivecpp";
        case implementation_type::nvhpc:
            return out << "nvhpc";
        case implementation_type::intel_llvm:
            return out << "intel_llvm";
        case implementation_type::gnu_tbb:
            return out << "gnu_tbb";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, implementation_type &impl) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "adaptivecpp" || str == "acpp") {
        impl = implementation_type::adaptivecpp;
    } else if (str == "nvhpc" || str == "nvcpp" || str == "nvc++") {
        impl = implementation_type::nvhpc;
    } else if (str == "intel_llvm" || str == "icpx" || str == "dpcpp" || str == "dpc++") {
        impl = implementation_type::intel_llvm;
    } else if (str == "gnu_tbb" || str == "gcc_tbb" || str == "g++_tbb" || str == "gnu" || str == "gcc" || str == "g++") {
        impl = implementation_type::gnu_tbb;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::stdpar

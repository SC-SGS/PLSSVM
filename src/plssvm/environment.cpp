/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/environment.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::environment {

std::ostream &operator<<(std::ostream &out, const status s) {
    switch (s) {
        case status::uninitialized:
            return out << "uninitialized";
        case status::initialized:
            return out << "initialized";
        case status::finalized:
            return out << "finalized";
        case status::unnecessary:
            return out << "unnecessary";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, status &s) {
    std::string str;
    in >> str;
    ::plssvm::detail::to_lower_case(str);

    if (str == "uninitialized") {
        s = status::uninitialized;
    } else if (str == "initialized") {
        s = status::initialized;
    } else if (str == "finalized") {
        s = status::finalized;
    } else if (str == "unnecessary") {
        s = status::unnecessary;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::environment

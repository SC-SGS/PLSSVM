/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/file_format_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>                               // std::ios::failbit
#include <istream>                           // std::istream
#include <ostream>                           // std::ostream
#include <string>                            // std::string

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const file_format_type format) {
    switch (format) {
        case file_format_type::libsvm:
            return out << "libsvm";
        case file_format_type::arff:
            return out << "arff";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, file_format_type &format) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "libsvm") {
        format = file_format_type::libsvm;
    } else if (str == "arff") {
        format = file_format_type::arff;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm
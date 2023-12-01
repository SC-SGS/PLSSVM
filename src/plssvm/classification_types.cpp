/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/classification_types.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <cstddef>      // std::size_t
#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const classification_type classification) {
    switch (classification) {
        case classification_type::oaa:
            return out << "oaa";
        case classification_type::oao:
            return out << "oao";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, classification_type &classification) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "oaa" || str == "one_vs_all" || str == "one_against_all") {
        classification = classification_type::oaa;
    } else if (str == "oao" || str == "one_vs_one" || str == "one_against_one") {
        classification = classification_type::oao;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

std::string_view classification_type_to_full_string(const classification_type classification) {
    switch (classification) {
        case classification_type::oaa:
            return "one vs. all";
        case classification_type::oao:
            return "one vs. one";
    }
    return "unknown";
}

std::size_t calculate_number_of_classifiers(const classification_type classification, const std::size_t num_classes) {
    PLSSVM_ASSERT(num_classes >= 2, "At least two classes must be given!");

    switch (classification) {
        case classification_type::oaa:
            return num_classes;
        case classification_type::oao:
            return num_classes * (num_classes - 1) / 2;
    }
    return 0;
}

}  // namespace plssvm
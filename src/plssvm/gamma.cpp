/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/gamma.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "fmt/core.h"  // fmt::format

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <sstream>  // std::istringstream
#include <string>   // std::string
#include <variant>  // std::variant, std::visit, std::holds_alternative

namespace plssvm {

std::string get_gamma_string(const gamma_type &var) {
    return std::visit(detail::overloaded{
                          [](const real_type val) { return fmt::format("{}", val); },
                          [&](const gamma_coefficient_type val) -> std::string {
                              switch (val) {
                                  case gamma_coefficient_type::automatic:
                                      return "\"1 / num_features\"";
                                  case gamma_coefficient_type::scale:
                                      return "\"1 / (num_features * variance(input_data))\"";
                              }
                          } },
                      var);
}

std::ostream &operator<<(std::ostream &out, const gamma_coefficient_type gamma_coefficient) {
    switch (gamma_coefficient) {
        case gamma_coefficient_type::automatic:
            return out << "automatic";
        case gamma_coefficient_type::scale:
            return out << "scale";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, gamma_coefficient_type &gamma_coefficient) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic" || str == "auto") {
        gamma_coefficient = gamma_coefficient_type::automatic;
    } else if (str == "scale") {
        gamma_coefficient = gamma_coefficient_type::scale;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

std::ostream &operator<<(std::ostream &out, const gamma_type gamma_value) {
    std::visit(detail::overloaded{
                   [&](const real_type val) { out << val; },
                   [&](const gamma_coefficient_type val) { out << val; } },
               gamma_value);
    return out;
}

std::istream &operator>>(std::istream &in, gamma_type &gamma_value) {
    // save current istream buffer in a string
    // -> necessary since we may need to parse the content twice
    std::string temp{};
    in >> temp;
    detail::to_lower_case(temp);
    temp = detail::trim(temp);

    // try parsing the value
    std::istringstream is{ temp };

    gamma_coefficient_type gamma_coef{};
    is >> gamma_coef;
    if (!is.fail()) {
        // read an enum type -> set active std::variant member accordingly
        gamma_value = gamma_coef;
    } else {
        // string couldn't be parsed -> reset stream and try again
        is = std::istringstream{ temp };

        real_type gamma_real{};
        is >> gamma_real;

        if (!is.fail()) {
            // read a real_type -> set active std::variant member accordingly
            gamma_value = gamma_real;
        } else {
            in.setstate(std::ios::failbit);
        }
    }
    return in;
}

}  // namespace plssvm

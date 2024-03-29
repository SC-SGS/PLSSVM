/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/kernel_function_types.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/parameter.hpp"              // plssvm::detail::parameter

#include "fmt/core.h"                        // fmt::format

#include <ios>                               // std::ios::failbit
#include <istream>                           // std::istream
#include <ostream>                           // std::ostream
#include <string>                            // std::string
#include <string_view>                       // std::string_view
#include <vector>                            // std::vector

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const kernel_function_type kernel) {
    switch (kernel) {
        case kernel_function_type::linear:
            return out << "linear";
        case kernel_function_type::polynomial:
            return out << "polynomial";
        case kernel_function_type::rbf:
            return out << "rbf";
    }
    return out << "unknown";
}

std::string_view kernel_function_type_to_math_string(const kernel_function_type kernel) noexcept {
    switch (kernel) {
        case kernel_function_type::linear:
            return "u'*v";
        case kernel_function_type::polynomial:
            return "(gamma*u'*v+coef0)^degree";
        case kernel_function_type::rbf:
            return "exp(-gamma*|u-v|^2)";
    }
    return "unknown";
}

std::istream &operator>>(std::istream &in, kernel_function_type &kernel) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "linear" || str == "0") {
        kernel = kernel_function_type::linear;
    } else if (str == "polynomial" || str == "1") {
        kernel = kernel_function_type::polynomial;
    } else if (str == "rbf" || str == "2") {
        kernel = kernel_function_type::rbf;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

template <typename real_type>
real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj, const detail::parameter<real_type> &params) {
    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            return kernel_function<kernel_function_type::linear>(xi, xj);
        case kernel_function_type::polynomial:
            return kernel_function<kernel_function_type::polynomial>(xi, xj, params.degree, params.gamma, params.coef0);
        case kernel_function_type::rbf:
            return kernel_function<kernel_function_type::rbf>(xi, xj, params.gamma);
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(params.kernel_type)) };
}

template float kernel_function(const std::vector<float> &, const std::vector<float> &, const detail::parameter<float> &);
template double kernel_function(const std::vector<double> &, const std::vector<double> &, const detail::parameter<double> &);

}  // namespace plssvm
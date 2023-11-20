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
#include "plssvm/matrix.hpp"                 // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"  // fmt::format

#include <cstddef>      // std::Size_t
#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

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
    } else if (str == "polynomial" || str == "poly" || str == "1") {
        kernel = kernel_function_type::polynomial;
    } else if (str == "rbf" || str == "2") {
        kernel = kernel_function_type::rbf;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

template <typename T>
T kernel_function(const std::vector<T> &xi, const std::vector<T> &xj, const parameter &params) {
    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            return kernel_function<kernel_function_type::linear>(xi, xj);
        case kernel_function_type::polynomial:
            return kernel_function<kernel_function_type::polynomial>(xi, xj, params.degree, static_cast<T>(params.gamma), static_cast<T>(params.coef0));
        case kernel_function_type::rbf:
            return kernel_function<kernel_function_type::rbf>(xi, xj, static_cast<T>(params.gamma));
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(params.kernel_type)) };
}

template float kernel_function(const std::vector<float> &, const std::vector<float> &, const parameter &);
template double kernel_function(const std::vector<double> &, const std::vector<double> &, const parameter &);

template <typename T, layout_type layout>
T kernel_function(const matrix<T, layout> &x, const std::size_t i, const matrix<T, layout> &y, const std::size_t j, const parameter &params) {
    PLSSVM_ASSERT(x.num_cols() == y.num_cols(), "Sizes mismatch!: {} != {}", x.num_cols(), y.num_cols());
    PLSSVM_ASSERT(i < x.num_rows(), "Out-of-bounce access for i and x!: {} < {}", i, x.num_rows());
    PLSSVM_ASSERT(j < y.num_rows(), "Out-of-bounce access for j and y!: {} < {}", j, y.num_rows());

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            return kernel_function<kernel_function_type::linear>(x, i, y, j);
        case kernel_function_type::polynomial:
            return kernel_function<kernel_function_type::polynomial>(x, i, y, j, params.degree.value(), static_cast<T>(params.gamma.value()), static_cast<T>(params.coef0.value()));
        case kernel_function_type::rbf:
            return kernel_function<kernel_function_type::rbf>(x, i, y, j, static_cast<T>(params.gamma.value()));
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(params.kernel_type)) };
}

template float kernel_function(const matrix<float, layout_type::aos> &, const std::size_t, const matrix<float, layout_type::aos> &, const std::size_t, const parameter &);
template float kernel_function(const matrix<float, layout_type::soa> &, const std::size_t, const matrix<float, layout_type::soa> &, const std::size_t, const parameter &);
template double kernel_function(const matrix<double, layout_type::aos> &, const std::size_t, const matrix<double, layout_type::aos> &, const std::size_t, const parameter &);
template double kernel_function(const matrix<double, layout_type::soa> &, const std::size_t, const matrix<double, layout_type::soa> &, const std::size_t, const parameter &);


}  // namespace plssvm
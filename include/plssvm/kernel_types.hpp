/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all available kernel types.
 */

#pragma once

#include "plssvm/detail/operators.hpp"

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <algorithm>    // std::transform
#include <cassert>      // assert
#include <cctype>       // std::tolower
#include <cmath>        // std::pow, std::exp
#include <cstddef>      // std:size_t
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string_view>  // std::string_view
#include <tuple>        // std::forward_as_tuple, std::get
#include <utility>      // std::forward

namespace plssvm {

/**
 * @brief Enum class for the different kernel types.
 */
enum class kernel_type {
    /**  \f$\vec{u} \cdot \vec{v}\f$ */
    linear = 0,
    /** \f$(gamma \cdot \vec{u} \cdot \vec{v} + coef0)^{degree}\f$ */
    polynomial = 1,
    /** \f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$ */
    rbf = 2
};

/**
 * @brief Stream-insertion-operator overload for convenient printing of the kernel type @p kernel.
 * @param[inout] out the output-stream to write the kernel type to
 * @param[in] kernel the kernel type
 * @return the output-stream
 */
inline std::ostream &operator<<(std::ostream &out, const kernel_type kernel) {
    switch (kernel) {
        case kernel_type::linear:
            return out << "linear";
        case kernel_type::polynomial:
            return out << "polynomial";
        case kernel_type::rbf:
            return out << "rbf";
        default:
            return out << "unknown";
    }
}

/**
 * @brief Stream-extraction-operator overload for convenient converting a string to a kernel type.
 * @param[inout] in input-stream to extract the kernel type from
 * @param[in] kernel the kernel type
 * @return the input-stream
 */
inline std::istream &operator>>(std::istream &in, kernel_type &kernel) {
    std::string str;
    in >> str;
    std::transform(str.begin(), str.end(), str.begin(), [](const char c) { return std::tolower(c); });

    if (str == "linear" || str == "0") {
        kernel = kernel_type::linear;
    } else if (str == "polynomial" || str == "1") {
        kernel = kernel_type::polynomial;
    } else if (str == "rbf" || str == "2") {
        kernel = kernel_type::rbf;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

namespace detail {
template <std::size_t I, class... Ts>
decltype(auto) get(Ts &&...ts) {
    return std::get<I>(std::forward_as_tuple(ts...));
}

template <typename>
constexpr bool always_false_v = false;
}  // namespace detail

template <kernel_type kernel, typename real_type, typename size_type, typename... Args>
real_type kernel_function(const real_type *xi, const real_type *xj, const size_type dim, Args &&...args) {
    if constexpr (kernel == kernel_type::linear) {
        static_assert(sizeof...(args) == 0, "Illegal number of additional parameters!");
        return mult(xi, xj, dim);
    } else if constexpr (kernel == kernel_type::polynomial) {
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters!");
        auto degree = detail::get<0>(args...);
        auto gamma = detail::get<1>(args...);
        auto coef0 = detail::get<2>(args...);
        return std::pow(gamma * mult(xi, xj, dim) + coef0, degree);
    } else if constexpr (kernel == kernel_type::rbf) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters!");
        auto gamma = detail::get<0>(args...);
        real_type temp = 0.0;
        for (size_type i = 0; i < dim; ++i) {
            temp += xi[i] - xj[i];
        }
        return std::exp(-gamma * temp * temp);
    } else {
        static_assert(detail::always_false_v<real_type>, "Unknown kernel type!");
    }
}
template <kernel_type kernel, typename real_type, typename... Args>
real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj, Args &&...args) {
    assert((xi.size() == xj.size()) && "Sizes in kernel function mismatch!");
    return kernel_function<kernel>(xi.data(), xj.data(), xi.size(), std::forward<Args>(args)...);
}

}  // namespace plssvm

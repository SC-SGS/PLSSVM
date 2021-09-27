/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all available kernel types.
 */

#pragma once

#include "plssvm/detail/assert.hpp"     // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"  // dot product, plssvm::squared_euclidean_dist
#include "plssvm/detail/utility.hpp"    // plssvm::detail::always_false_v

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <algorithm>  // std::transform
#include <cctype>     // std::tolower
#include <cmath>      // std::pow, std::exp
#include <istream>    // std::istream
#include <ostream>    // std::ostream
#include <string>     // std::string

namespace plssvm {

/**
 * @brief Enum class for the different kernel types.
 */
enum class kernel_type {
    /**  \f$\vec{u}^T \cdot \vec{v}\f$ */
    linear = 0,
    /** \f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$ */
    polynomial = 1,
    /** \f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$ */
    rbf = 2
};

/**
 * @brief Stream-insertion operator overload for convenient printing of the kernel type @p kernel.
 * @param[in,out] out the output-stream to write the kernel type to
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
    }
    return out << "unknown";
}

/**
 * @brief Stream-extraction operator overload for convenient converting a string to a kernel type.
 * @details The extracted value is matched case-insensitive and can be the integer value of the kernel_type.
 * @param[in,out] in input-stream to extract the kernel type from
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

/**
 * @brief Computes the value of the two vectors @p xi and @p xj using the kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam real_type the type of the values
 * @tparam Args additional parameters used in the respective kernel functions
 * @param[in] xi the first vector
 * @param[in] xj the second vector
 * @param[in] args additional parameters
 * @return the value computed by the kernel function
 */
template <kernel_type kernel, typename real_type, typename... Args>
real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj, Args &&...args) {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    if constexpr (kernel == kernel_type::linear) {
        static_assert(sizeof...(args) == 0, "Illegal number of additional parameters! Must be 0.");
        return transposed{ xi } * xj;
    } else if constexpr (kernel == kernel_type::polynomial) {
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
        const auto degree = static_cast<real_type>(detail::get<0>(args...));
        const auto gamma = static_cast<real_type>(detail::get<1>(args...));
        const auto coef0 = static_cast<real_type>(detail::get<2>(args...));
        return std::pow(std::fma(gamma, (transposed<real_type>{ xi } * xj), coef0), static_cast<real_type>(degree));
    } else if constexpr (kernel == kernel_type::rbf) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma = static_cast<real_type>(detail::get<0>(args...));
        return std::exp(-gamma * squared_euclidean_dist(xi, xj));
    } else {
        static_assert(detail::always_false_v<real_type>, "Unknown kernel type!");
    }
}

}  // namespace plssvm

/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all available kernel types.
 */

#pragma once

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <algorithm>    // std::transform
#include <cctype>       // std::tolower
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string_view>  // std::string_view

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

}  // namespace plssvm

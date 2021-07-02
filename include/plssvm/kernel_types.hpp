/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all available kernel types.
 */

#pragma once

#include <fmt/ostream.h>

#include <ostream>
#include <string_view>

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
 * @brief Output-operator overload for convenient printing of the kernel type @p kernel.
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

}  // namespace plssvm

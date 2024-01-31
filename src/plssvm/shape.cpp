/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/shape.hpp"

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::swap
#include <iostream>   // std:ostream, std::istream

namespace plssvm {

shape::shape(const std::size_t x_p, const std::size_t y_p) noexcept :
    x{ x_p },
    y{ y_p } { }

void shape::swap(shape &other) noexcept {
    using std::swap;
    swap(this->x, other.x);
    swap(this->y, other.y);
}

std::ostream &operator<<(std::ostream &out, const shape s) {
    return out << fmt::format("[{}, {}]", s.x, s.y);
}

std::istream &operator>>(std::istream &in, shape &s) {
    return in >> s.x >> s.y;
}

void swap(shape &lhs, shape &rhs) noexcept {
    lhs.swap(rhs);
}

bool operator==(shape lhs, shape rhs) noexcept {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool operator!=(shape lhs, shape rhs) noexcept {
    return !(lhs == rhs);
}

}  // namespace plssvm

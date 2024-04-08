/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a struct encapsulating a memory size as well as custom literals regarding byte units.
 */

#ifndef PLSSVM_DETAIL_MEMORY_SIZE_HPP_
#define PLSSVM_DETAIL_MEMORY_SIZE_HPP_
#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "fmt/core.h"     // fmt::format, fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstddef>     // std::size_t
#include <functional>  // std::hash
#include <iosfwd>      // forward declare std::ostream and std::istream

namespace plssvm::detail {

/**
 * @brief A class encapsulating a memory size in bytes.
 * @details Mainly used as result type for the custom literals like `2_MB`.
 */
class memory_size {
  public:
    /**
     * @brief Construct a new memory size representing zero bytes.
     */
    constexpr memory_size() = default;

    /**
     * @brief Construct a new memory size representing @p val bytes.
     * @param[in] val the number of bytes
     */
    explicit constexpr memory_size(const unsigned long long val) noexcept :
        size_in_bytes_{ val } { }

    /**
     * @brief Return the number of bytes stored in this memory size object.
     * @return the number of bytes (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr unsigned long long num_bytes() const noexcept {
        return size_in_bytes_;
    }

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other memory size
     */
    constexpr void swap(memory_size &other) noexcept {
        // can't use std::swap since it isn't constexpr in C++17
        const unsigned long long temp{ other.size_in_bytes_ };
        other.size_in_bytes_ = size_in_bytes_;
        size_in_bytes_ = temp;
    }

  private:
    /// The number of bytes.
    unsigned long long size_in_bytes_{ 0ULL };
};

/**
 * @brief Swap the contents of @p lhs with the contents of @p rhs.
 * @param[in,out] lhs the first memory size
 * @param[in,out] rhs the second memory size
 */
constexpr void swap(memory_size &lhs, memory_size &rhs) noexcept {
    lhs.swap(rhs);
}

/**
 * @brief Add the memory size @p rhs to @p lhs.
 * @param[in,out] lhs the memory size to add @p rhs to
 * @param[in] rhs the memory size to add
 * @return a reference to the modified memory size @p lhs
 */
constexpr memory_size &operator+=(memory_size &lhs, const memory_size rhs) {
    lhs = memory_size{ lhs.num_bytes() + rhs.num_bytes() };
    return lhs;
}

/**
 * @brief Add two memory sizes together.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return the sum of both memory sizes (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr memory_size operator+(memory_size lhs, const memory_size rhs) {
    lhs += rhs;
    return lhs;
}

/**
 * @brief Subtract the memory size @p rhs from @p lhs.
 * @details The users has to take care, that @p rhs < @p lhs, since negative values can't be represented in a `plssvm::detail::memory_size`.
 * @param[in,out] lhs the memory size to subtract @p rhs from
 * @param[in] rhs the memory size to subtract
 * @return a reference to the modified memory size @p lhs
 */
constexpr memory_size &operator-=(memory_size &lhs, const memory_size rhs) {
    lhs = memory_size{ lhs.num_bytes() - rhs.num_bytes() };
    return lhs;
}

/**
 * @brief Subtract two memory sizes from each other.
 * @details The users has to take care, that @p rhs < @p lhs, since negative values can't be represented in a `plssvm::detail::memory_size`.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return the difference of both memory sizes (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr memory_size operator-(memory_size lhs, const memory_size rhs) {
    lhs -= rhs;
    return lhs;
}

/**
 * @brief Scale the provided memory size @p mem by the given @p factor.
 * @param[in] mem the memory size to scale
 * @param[in] factor the scaling factor
 * @return a reference to the modified memory size @p mem
 */
constexpr memory_size &operator*=(memory_size &mem, const long double factor) {
    mem = memory_size{ static_cast<unsigned long long>(mem.num_bytes() * factor) };
    return mem;
}

/**
 * @brief Scale the provided memory size @p mem by the given @p factor.
 * @param[in] mem the memory size
 * @param[in] factor the scaling factor
 * @return the scaled memory size (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr memory_size operator*(memory_size mem, const long double factor) {
    mem *= factor;
    return mem;
}

/**
 * @copydoc plssvm::detail::operator*(const memory_size, const long double)
 */
[[nodiscard]] constexpr memory_size operator*(const long double factor, memory_size mem) {
    mem *= factor;
    return mem;
}

/**
 * @copydoc plssvm::detail::operator*=(memory_size &, const long double)
 */
constexpr memory_size &operator/=(memory_size &mem, const long double factor) {
    mem = memory_size{ static_cast<unsigned long long>(mem.num_bytes() / factor) };
    return mem;
}

/**
 * @copydoc plssvm::detail::operator*(const memory_size, const long double)
 */
[[nodiscard]] constexpr memory_size operator/(memory_size mem, const long double factor) {
    mem /= factor;
    return mem;
}

/**
 * @brief Calculate the fraction between two memory sizes.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return the factor (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr long double operator/(const memory_size lhs, const memory_size rhs) {
    return static_cast<long double>(lhs.num_bytes()) / static_cast<long double>(rhs.num_bytes());
}

/**
 * @brief Compare two memory sizes for equality.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return `true` if both encapsulated memory sizes are equal, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator==(const memory_size lhs, const memory_size rhs) {
    return lhs.num_bytes() == rhs.num_bytes();
}

/**
 * @brief Compare two memory sizes for inequality.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return `false` if both encapsulated memory sizes are equal, otherwise `true` (`[[nodiscard]]`)
 */
constexpr bool operator!=(const memory_size lhs, const memory_size rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Check whether the memory size @p lhs is less than @p rhs.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return `true` if the encapsulated memory sizes of @p lhs is smaller than the one of @p rhs, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator<(const memory_size lhs, const memory_size rhs) {
    return lhs.num_bytes() < rhs.num_bytes();
}

/**
 * @brief Check whether the memory size @p lhs is greater than @p rhs.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return `true` if the encapsulated memory sizes of @p lhs is greater than the one of @p rhs, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator>(const memory_size lhs, const memory_size rhs) {
    return lhs.num_bytes() > rhs.num_bytes();
}

/**
 * @brief Check whether the memory size @p lhs is less or equal than @p rhs.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return `true` if the encapsulated memory sizes of @p lhs is smaller or equal than the one of @p rhs, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator<=(const memory_size lhs, const memory_size rhs) {
    return !(lhs > rhs);
}

/**
 * @brief Check whether the memory size @p lhs is greater or equal than @p rhs.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return `true` if the encapsulated memory sizes of @p lhs is greater or equal than the one of @p rhs, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator>=(const memory_size lhs, const memory_size rhs) {
    return !(lhs < rhs);
}

/**
 * @brief Output the memory size @p mem to the given output-stream @p out.
 * @details Also outputs the biggest possible binary memory unit such that the memory size has at least one digit before the decimal delimiter,
 *          e.g., `memory_size{ 2048 }` will output "2.00 GiB".
 * @param[in,out] out the output-stream to write the memory size to
 * @param[in] mem the memory size
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, memory_size mem);

/**
 * @brief Use the input-stream @p in to initialize the @p mem memory size.
 * @details Example inputs are `"1B"`, `"2.0 KiB"`, or `"3.5 MB"`.
 * @param[in,out] in input-stream to extract the memory size from
 * @param[in] mem the memory size
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, memory_size &mem);

//*************************************************************************************************************************************//
//                                                           custom literals                                                           //
//*************************************************************************************************************************************//
/**
 * @brief Constexpr friendly power function. Computes: `base^exponent`.
 * @details Only positive @p base and @p exponent values are allowed.
 * @param[in] base the base
 * @param[in] exponent the exponent
 * @return the power function (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr unsigned long long constexpr_pow(const unsigned long long base, const unsigned long long exponent) {
    unsigned long long ret{ 1 };
    for (unsigned long long e = 0; e < exponent; ++e) {
        ret *= base;
    }
    return ret;
}

namespace literals {

/**
 * @def PLSSVM_DEFINE_MEMORY_SIZE_LITERAL
 * @brief Defines a macro to create custom literals for different memory sizes for `long double` and `unsigned long long`.
 * @param[in] name the name of the custom literal
 * @param[in] factor either `1000` for decimal prefixes (e.g., KB) or `1024` for binary  prefixes (e.g., KiB)
 * @param[in] power the magnitude, e.g., KB (1) or MB (2)
 */
#define PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(name, factor, power)                                                                                                    \
    constexpr memory_size operator""_##name(const long double val) { return memory_size{ static_cast<unsigned long long>(val * constexpr_pow(factor, power)) }; } \
    constexpr memory_size operator""_##name(const unsigned long long val) { return memory_size{ val * constexpr_pow(factor, power) }; }

// byte
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(B, 1000, 0)

// decimal prefixes
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(KB, 1000, 1)
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(MB, 1000, 2)
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(GB, 1000, 3)
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(TB, 1000, 4)

// binary prefixes
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(KiB, 1024, 1)
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(MiB, 1024, 2)
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(GiB, 1024, 3)
PLSSVM_DEFINE_MEMORY_SIZE_LITERAL(TiB, 1024, 4)

#undef PLSSVM_DEFINE_MEMORY_SIZE_LITERAL

}  // namespace literals

}  // namespace plssvm::detail

namespace std {

/**
 * @brief Hashing struct specialization in the `std` namespace for a `plssvm::detail::memory_size`.
 * @details Necessary to be able to use a memory_size, e.g., in a `std::unordered_map`.
 */
template <>
struct hash<plssvm::detail::memory_size> {
    /**
     * @brief Overload the function call operator for a memory_size.
     * @param[in] mem the memory_size to hash
     * @return the hash value of @p mem
     */
    std::size_t operator()(const plssvm::detail::memory_size &mem) const noexcept {
        return std::hash<unsigned long long>{}(mem.num_bytes());
    }
};

}  // namespace std

template <>
struct fmt::formatter<plssvm::detail::memory_size> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_MEMORY_SIZE_HPP_

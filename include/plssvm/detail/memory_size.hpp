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

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm::detail {

/**
 * @brief A class encapsulating a memory size in bytes.
 */
class memory_size {
  public:
    /**
     * @brief Construct a new memory size with zero bytes.
     */
    constexpr memory_size() = default;
    /**
     * @brief Construct a new memory size representing @p val bytes.
     * @param[in] val the number of bytes
     */
    explicit constexpr memory_size(const unsigned long long val) noexcept :
        size_in_bytes_{ val } {}

    /**
     * @brief Return the number of bytes stored in this memory size object.
     * @return the number of bytes (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr unsigned long long num_bytes() const noexcept {
        return size_in_bytes_;
    }

  private:
    /// The number of bytes.
    unsigned long long size_in_bytes_{ 0ULL };
};

/**
 * @brief Add two memory sizes to each other.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return the sum of both memory sizes (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr memory_size operator+(const memory_size lhs, const memory_size rhs) {
    return memory_size{ lhs.num_bytes() + rhs.num_bytes() };
}
/**
 * @brief Subtract two memory sizes from each other.
 * @details The users has to take care, that rhs < lhs, since negative values can't be represented in a plssvm::detail::memory_size.
 * @param[in] lhs the first memory size
 * @param[in] rhs the second memory size
 * @return the difference of both memory sizes (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr memory_size operator-(const memory_size lhs, const memory_size rhs) {
    return memory_size{ lhs.num_bytes() - rhs.num_bytes() };
}
/**
 * @brief Scale the provided memory size by the given @p factor.
 * @param[in] mem the memory size
 * @param[in] factor the scaling factor
 * @return the scaled memory sizes (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr memory_size operator*(const memory_size mem, long double factor) {
    return memory_size{ static_cast<unsigned long long>(mem.num_bytes() * factor) };
}
/**
 * @copydoc plssvm::detail::operator*(const memory_size, long double)
 */
[[nodiscard]] constexpr memory_size operator*(long double factor, const memory_size mem) {
    return memory_size{ static_cast<unsigned long long>(mem.num_bytes() * factor) };
}
/**
 * @copydoc plssvm::detail::operator*(const memory_size, long double)
 */
[[nodiscard]] constexpr memory_size operator/(const memory_size mem, long double factor) {
    return memory_size{ static_cast<unsigned long long>(mem.num_bytes() / factor) };
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
 * @brief Output the @p mem to the given output-stream @p out.
 * @details Also outputs the biggest possible memory unit such that the memory size has at least one digit before the decimal delimiter,
 *          e.g., `memory_size{ 2048 }` will output "2.00 GiB".
 * @param[in,out] out the output-stream to write the classification type to
 * @param[in] mem the encapsulated memory size
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, memory_size mem);

/**
 * @brief Use the input-stream @p in to initialize the @p memory size.
 * @details Example inputs are "1B", "2.0 KiB", or "3.5 MB".
 * @param[in,out] in input-stream to extract the memory size from
 * @param[in] mem the memory size
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, memory_size &mem);

//*************************************************************************************************************************************//
//                                                           custom literals                                                           //
//*************************************************************************************************************************************//
namespace literals {

/// Convert bytes to bytes.
constexpr memory_size operator""_B(const long double val) { return memory_size{ static_cast<unsigned long long>(val) }; }
/// Convert bytes to bytes.
constexpr memory_size operator""_B(const unsigned long long val) { return memory_size{ val }; }

//*************************************************************************************************************************************//
//                                                    decimal prefix - long double                                                     //
//*************************************************************************************************************************************//

/// Convert 1 KB to bytes (factor 1'000).
constexpr memory_size operator""_KB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1000L) }; }
/// Convert 1 MB to bytes (factor 1'000'000).
constexpr memory_size operator""_MB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1000L * 1000L) }; }
/// Convert 1 GB to bytes (factor 1'000'000'000).
constexpr memory_size operator""_GB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1000L * 1000L * 1000L) }; }
/// Convert 1 TB to bytes (factor 1'000'000'000'000).
constexpr memory_size operator""_TB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1000L * 1000L * 1000L * 1000L) }; }

//*************************************************************************************************************************************//
//                                                 decimal prefix - unsigned long long                                                 //
//*************************************************************************************************************************************//

/// Convert 1 KB to bytes (factor 1'000).
constexpr memory_size operator""_KB(const unsigned long long val) { return memory_size{ val * 1000ULL }; }
/// Convert 1 MB to bytes (factor 1'000'000).
constexpr memory_size operator""_MB(const unsigned long long val) { return memory_size{ val * 1000ULL * 1000ULL }; }
/// Convert 1 GB to bytes (factor 1'000'000'000).
constexpr memory_size operator""_GB(const unsigned long long val) { return memory_size{ val * 1000ULL * 1000ULL * 1000ULL }; }
/// Convert 1 TB to bytes (factor 1'000'000'000'000).
constexpr memory_size operator""_TB(const unsigned long long val) { return memory_size{ val * 1000ULL * 1000ULL * 1000ULL * 1000ULL }; }

//*************************************************************************************************************************************//
//                                                     binary prefix - long double                                                     //
//*************************************************************************************************************************************//

/// Convert 1 KiB to bytes (factor 1'024).
constexpr memory_size operator""_KiB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1024L) }; }
/// Convert 1 MiB to bytes (factor 1'048'576).
constexpr memory_size operator""_MiB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1024L * 1024L) }; }
/// Convert 1 GiB to bytes (factor 1'073'741'824).
constexpr memory_size operator""_GiB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1024L * 1024L * 1024L) }; }
/// Convert 1 TiB to bytes (factor 1'099'511'627'776).
constexpr memory_size operator""_TiB(const long double val) { return memory_size{ static_cast<unsigned long long>(val * 1024L * 1024L * 1024L * 1024L) }; }

//*************************************************************************************************************************************//
//                                                 binary prefix - unsigned long long                                                  //
//*************************************************************************************************************************************//

/// Convert 1 KiB to bytes (factor 1'024).
constexpr memory_size operator""_KiB(const unsigned long long val) { return memory_size{ val * 1024ULL }; }
/// Convert 1 MiB to bytes (factor 1'048'576).
constexpr memory_size operator""_MiB(const unsigned long long val) { return memory_size{ val * 1024ULL * 1024ULL }; }
/// Convert 1 GiB to bytes (factor 1'073'741'824).
constexpr memory_size operator""_GiB(const unsigned long long val) { return memory_size{ val * 1024ULL * 1024ULL * 1024ULL }; }
/// Convert 1 TiB to bytes (factor 1'099'511'627'776).
constexpr memory_size operator""_TiB(const unsigned long long val) { return memory_size{ val * 1024ULL * 1024ULL * 1024ULL * 1024ULL }; }

}  // namespace literals

}  // namespace plssvm::detail

template <>
struct fmt::formatter<plssvm::detail::memory_size> : fmt::ostream_formatter {};

#endif  // PLSSVM_DETAIL_MEMORY_SIZE_HPP_

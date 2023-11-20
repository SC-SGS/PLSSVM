/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implementation of the SHA2-256 hashing algorithm.
 */

#ifndef PLSSVM_DETAIL_SHA256_HPP_
#define PLSSVM_DETAIL_SHA256_HPP_
#pragma once

#include <array>        // std::array
#include <cstdint>      // std::uint32_t
#include <string>       // std::string
#include <type_traits>  // std::enable_if_t, std::is_unsigned_v

namespace plssvm::detail {

/**
 * @brief A hashing struct used for sha256 hashing.
 * @details Based on: https://en.wikipedia.org/wiki/SHA-2#Pseudocode
 */
class sha256 {
  public:
    /**
     * @brief Calculate the sha256 hash for the @p input string.
     * @param[in] input the string to hash
     * @return the sha256 hash of @p input (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string operator()(std::string input) const;

  private:
    /**
     * @brief Unpack the bits of @p x into the @p str.
     * @tparam T the **unsigned** type and, therefore, number of bits, to unpack
     * @param[in] x the integer representing the bits to unpack
     * @param[out] str the string to unpack the bits to
     */
    template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    static void unpack(const T x, unsigned char *str) {
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            str[i] = static_cast<unsigned char>(x >> ((sizeof(T) - i - 1) * 8));
        }
    }
    /**
     * @brief Pack four byte of the @p str into the 32-bit unsigned integer @p x.
     * @param[in] str the string to pack
     * @param[out] x the 32-bit unsigned integer to pack the bytes to
     */
    static void pack32(const unsigned char *str, std::uint32_t &x);
    /**
     * @brief Rotate the bits in @p value @ count times to the right.
     * @details Based on: https://en.wikipedia.org/wiki/Circular_shift
     * @param[in] value the 32-bit integer to rotate
     * @param[in] count the number of bits to rotate
     * @return the rotated 32-bit integer (`[[nodiscard]]`)
     */
    [[nodiscard]] static std::uint32_t rotr32(std::uint32_t value, int count);

    /// Number of bytes in the resulting digest.
    static constexpr std::uint32_t DIGEST_SIZE = 256 / 8;
    /// NUmber of bytes processed in one round (chunk).
    static constexpr std::uint32_t CHUNK_SIZE = 512 / 8;

    /**
     * @brief Array of the sha256 round constants
     * @details First 32 bits of the fractional parts of the cube roots of the first 64 primes 2..311.
     */
    static constexpr std::array<std::uint32_t, 64> k_{
        // clang-format off
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        // clang-format on
    };
};

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_SHA256_HPP_
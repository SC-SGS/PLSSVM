/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/sha256.hpp"

#include "fmt/format.h"  // fmt::format, fmt::join

#include <array>    // std::array
#include <cstdint>  // std::uint32_t, std::uint64_t
#include <limits>   // std::numeric_limits::digits
#include <string>   // std::string

namespace plssvm::detail {

std::string sha256::operator()(std::string input) const {
    // Initialize hash values: (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
    std::array<std::uint32_t, 8> hash_values = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

    // Pre-processing (Padding):

    // begin with the original message of length L byte
    const std::uint64_t L = input.size();
    const std::uint64_t L_bits = L * 8;

    // append a single '1' byte
    input.append(1, static_cast<char>(0x80));

    // append K '0' bytes, where K is the minimum number >= 0 such that (L + 1 + K + 8) is a multiple of 512 / 8 = 64
    const std::uint32_t K = CHUNK_SIZE - (L + 1 + 8) % CHUNK_SIZE;
    input.resize(L + 1 + K + 8);

    auto *input_unsigned_ptr = reinterpret_cast<unsigned char *>(input.data());

    // append L as an 8-byte big-endian integer, making the total post-processed length a multiple of 64 byte
    // such that the bits in the message are: <original message of length L> 1 <K zeros> <L as 8 byte integer> , (the number of bytes will be a multiple of 64)
    unpack(L_bits, input_unsigned_ptr + input.size() - sizeof(L_bits));

    // break message into 512-bit chunks
    for (std::string::size_type chunk = 0; chunk < input.size() / CHUNK_SIZE; ++chunk) {
        // create a 64-entry message schedule array w[0..63] of 32-bit words
        std::array<std::uint32_t, 64> w{};
        // copy chunk into first 16 words w[0..15] of the message schedule array
        for (int i = 0; i < 16; ++i) {
            pack32(input_unsigned_ptr + chunk * CHUNK_SIZE + i * sizeof(std::uint32_t), w[i]);
        }

        // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
        for (int i = 16; i < 64; ++i) {
            // s0 := (w[i-15] right rotate  7) xor (w[i-15] right rotate 18) xor (w[i-15] rightshift  3)
            const std::uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            // s1 := (w[i-2] right rotate 17) xor (w[i-2] right rotate 19) xor (w[i-2] rightshift 10)
            const std::uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            // w[i] := w[i-16] + s0 + w[i-7] + s1
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        // Initialize working variables to current hash value:
        // a := h0; b := h1; c := h2; d := h3; e := h4; f := h5; g := h6; h := h7
        std::array<std::uint32_t, 8> wv = hash_values;

        // Compression function main loop:
        for (int i = 0; i < 64; ++i) {
            // S1 := (e right rotate 6) xor (e right rotate 11) xor (e right rotate 25)
            const std::uint32_t S1 = rotr32(wv[4], 6) ^ rotr32(wv[4], 11) ^ rotr32(wv[4], 25);
            // ch := (e and f) xor ((not e) and g)
            const std::uint32_t ch = (wv[4] & wv[5]) ^ (~wv[4] & wv[6]);
            // temp1 := h + S1 + ch + k[i] + w[i]
            const std::uint32_t temp1 = wv[7] + S1 + ch + k_[i] + w[i];
            // S0 := (a right rotate 2) xor (a right rotate 13) xor (a right rotate 22)
            const std::uint32_t S0 = rotr32(wv[0], 2) ^ rotr32(wv[0], 13) ^ rotr32(wv[0], 22);
            // maj := (a and b) xor (a and c) xor (b and c)
            const std::uint32_t maj = (wv[0] & wv[1]) ^ (wv[0] & wv[2]) ^ (wv[1] & wv[2]);
            // temp2 := S0 + maj
            const std::uint32_t temp2 = S0 + maj;

            // h := g; g := f; f := e; e := d + temp1; d := c; c := b; b := a; a := temp1 + temp2
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + temp1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = temp1 + temp2;
        }

        // Add the compressed chunk to the current hash value:
        for (int i = 0; i < 8; ++i) {
            hash_values[i] += wv[i];
        }
    }

    // Produce the final hash value (big-endian):
    std::array<unsigned char, DIGEST_SIZE> digest{};
    for (int i = 0; i < 8; ++i) {
        unpack(hash_values[i], digest.data() + i * sizeof(std::uint32_t));
    }

    // convert digest to result string
    return fmt::format("{:02x}", fmt::join(digest, ""));
}

void sha256::pack32(const unsigned char *str, std::uint32_t &x) {
    x = static_cast<std::uint32_t>(str[3])
        | static_cast<std::uint32_t>(str[2] << 8)
        | static_cast<std::uint32_t>(str[1] << 16)
        | static_cast<std::uint32_t>(str[0] << 24);
}

std::uint32_t sha256::rotr32(const std::uint32_t value, int count) {
    // prevent UB if count is 0 or greater than sizeof(std::uint32_t)
    const unsigned int mask = std::numeric_limits<std::uint32_t>::digits - 1;
    count &= mask;
    return (value >> count) | (value << (-count & mask));
}

}  // namespace plssvm::detail
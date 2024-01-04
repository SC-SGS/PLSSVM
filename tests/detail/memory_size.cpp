/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the memory size class and custom literals.
 */

#include "plssvm/detail/memory_size.hpp"  // plssvm::detail::memory_size

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_FROM_STRING, EXPECT_CONVERSION_TO_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <algorithm>   // std::swap
#include <cstddef>     // std::size_t
#include <functional>  // std::hash
#include <sstream>     // std::istringstream
#include <string>      // std::string

TEST(MemorySize, default_construct) {
    // default construct a memory size object
    const plssvm::detail::memory_size mem{};

    // 0 bytes should be reported
    EXPECT_EQ(mem.num_bytes(), 0ULL);
}

TEST(MemorySize, construct_from_ull) {
    // construct a memory size object using an unsigned long long
    const plssvm::detail::memory_size mem{ 1024ULL };
    EXPECT_EQ(mem.num_bytes(), 1024ULL);
}

TEST(MemorySize, num_bytes) {
    // create memory size object
    const plssvm::detail::memory_size mem{ 512 };
    // check getter
    EXPECT_EQ(mem.num_bytes(), 512ULL);
}

TEST(MemorySize, member_swap) {
    // construct two memory sizes
    plssvm::detail::memory_size mem1{ 1024ULL };
    plssvm::detail::memory_size mem2{ 500ULL };

    // swap both memory sizes
    mem1.swap(mem2);

    // check whether the contents have been swapped correctly
    EXPECT_EQ(mem1.num_bytes(), 500ULL);
    EXPECT_EQ(mem2.num_bytes(), 1024ULL);
}

TEST(MemorySize, free_swap) {
    // construct two memory sizes
    plssvm::detail::memory_size mem1{ 1024ULL };
    plssvm::detail::memory_size mem2{ 500ULL };

    // swap both memory sizes
    using std::swap;
    swap(mem1, mem2);

    // check whether the contents have been swapped correctly
    EXPECT_EQ(mem1.num_bytes(), 500ULL);
    EXPECT_EQ(mem2.num_bytes(), 1024ULL);
}

TEST(MemorySize, operator_compound_add) {
    // create two memory size objects
    plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };

    // check addition
    mem1 += mem2;
    EXPECT_EQ(mem1.num_bytes(), 1536ULL);
    // mem2 should not have changed
    EXPECT_EQ(mem2.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_add) {
    // create two memory size objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };

    // check addition
    const plssvm::detail::memory_size mem = mem1 + mem2;
    EXPECT_EQ(mem.num_bytes(), 1536ULL);
    EXPECT_EQ(mem1.num_bytes(), 1024ULL);
    EXPECT_EQ(mem2.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_compound_subtract) {
    // create two memory size objects
    plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };

    // check addition
    mem1 -= mem2;
    EXPECT_EQ(mem1.num_bytes(), 512ULL);
    // mem2 should not have changed
    EXPECT_EQ(mem2.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_subtract) {
    // create two memory size objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };

    // check subtraction
    const plssvm::detail::memory_size mem = mem1 - mem2;
    EXPECT_EQ(mem.num_bytes(), 512ULL);
    EXPECT_EQ(mem1.num_bytes(), 1024ULL);
    EXPECT_EQ(mem2.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_compound_scale_mul) {
    // create a memory size object
    plssvm::detail::memory_size mem{ 1024ULL };

    // check scale via multiplication
    mem *= 0.5L;
    EXPECT_EQ(mem.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_scale_mul) {
    // create a memory size object
    const plssvm::detail::memory_size mem{ 1024ULL };

    // check scale via multiplication
    const plssvm::detail::memory_size mem_r1 = mem * 0.5L;
    EXPECT_EQ(mem_r1.num_bytes(), 512ULL);
    EXPECT_EQ(mem.num_bytes(), 1024ULL);
    const plssvm::detail::memory_size mem_r2 = 0.5L * mem;
    EXPECT_EQ(mem_r2.num_bytes(), 512ULL);
    EXPECT_EQ(mem.num_bytes(), 1024ULL);
}

TEST(MemorySize, operator_compound_scale_div) {
    // create a memory size object
    plssvm::detail::memory_size mem{ 1024ULL };

    // check scale via division
    mem /= 2.0L;
    EXPECT_EQ(mem.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_scale_div) {
    // create a memory size object
    const plssvm::detail::memory_size mem{ 1024ULL };

    // check scale via division
    const plssvm::detail::memory_size mem_r1 = mem / 2.0L;
    EXPECT_EQ(mem_r1.num_bytes(), 512ULL);
    EXPECT_EQ(mem.num_bytes(), 1024ULL);
}

TEST(MemorySize, operator_factor) {
    // create two memory size objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };

    // check factor calculation
    const long double factor = mem1 / mem2;
    EXPECT_EQ(factor, 2.0L);
}

TEST(MemorySize, relational_equal) {
    // create memory objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };
    const plssvm::detail::memory_size mem3{ 512ULL };

    // check for equality
    EXPECT_FALSE(mem1 == mem2);
    EXPECT_TRUE(mem2 == mem3);
    EXPECT_FALSE(mem3 == mem1);
}

TEST(MemorySize, relational_inequal) {
    // create memory objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };
    const plssvm::detail::memory_size mem3{ 512ULL };

    // check for equality
    EXPECT_TRUE(mem1 != mem2);
    EXPECT_FALSE(mem2 != mem3);
    EXPECT_TRUE(mem3 != mem1);
}

TEST(MemorySize, relational_less) {
    // create memory objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };
    const plssvm::detail::memory_size mem3{ 512ULL };

    // check for equality
    EXPECT_FALSE(mem1 < mem2);
    EXPECT_FALSE(mem2 < mem3);
    EXPECT_TRUE(mem3 < mem1);
}

TEST(MemorySize, relational_greater) {
    // create memory objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };
    const plssvm::detail::memory_size mem3{ 512ULL };

    // check for equality
    EXPECT_TRUE(mem1 > mem2);
    EXPECT_FALSE(mem2 > mem3);
    EXPECT_FALSE(mem3 > mem1);
}

TEST(MemorySize, relational_less_or_equal) {
    // create memory objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };
    const plssvm::detail::memory_size mem3{ 512ULL };

    // check for equality
    EXPECT_FALSE(mem1 <= mem2);
    EXPECT_TRUE(mem2 <= mem3);
    EXPECT_TRUE(mem3 <= mem1);
}

TEST(MemorySize, relational_greater_or_equal) {
    // create memory objects
    const plssvm::detail::memory_size mem1{ 1024ULL };
    const plssvm::detail::memory_size mem2{ 512ULL };
    const plssvm::detail::memory_size mem3{ 512ULL };

    // check for equality
    EXPECT_TRUE(mem1 >= mem2);
    EXPECT_TRUE(mem2 >= mem3);
    EXPECT_FALSE(mem3 >= mem1);
}

TEST(MemorySize, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 0ULL }, "0 B");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 8ULL }, "8 B");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1024ULL }, "1.00 KiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 2048ULL }, "2.00 KiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1'048'576ULL }, "1.00 MiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 3'145'728ULL }, "3.00 MiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1'073'741'824ULL }, "1.00 GiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 4'294'967'296ULL }, "4.00 GiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1'099'511'627'776ULL }, "1.00 TiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 5'497'558'138'880ULL }, "5.00 TiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1'125'899'906'842'624ULL }, "1024.00 TiB");
}

TEST(MemorySize, from_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_FROM_STRING("0 B", plssvm::detail::memory_size{ 0ULL });
    EXPECT_CONVERSION_FROM_STRING("8 B", plssvm::detail::memory_size{ 8ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00 KiB", plssvm::detail::memory_size{ 1024ULL });
    EXPECT_CONVERSION_FROM_STRING("0.5 KB", plssvm::detail::memory_size{ 500ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00 MiB", plssvm::detail::memory_size{ 1'048'576ULL });
    EXPECT_CONVERSION_FROM_STRING("0.4 MB", plssvm::detail::memory_size{ 400'000ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00 GiB", plssvm::detail::memory_size{ 1'073'741'824ULL });
    EXPECT_CONVERSION_FROM_STRING("0.3 GB", plssvm::detail::memory_size{ 300'000'000ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00TiB", plssvm::detail::memory_size{ 1'099'511'627'776ULL });
    EXPECT_CONVERSION_FROM_STRING("0.2TB", plssvm::detail::memory_size{ 200'000'000'000ULL });
}

TEST(MemorySize, from_string_unknown) {
    // foo isn't a valid backend_type
    std::istringstream input{ "0Bit" };
    plssvm::detail::memory_size mem{};
    input >> mem;
    EXPECT_TRUE(input.fail());
}

TEST(MemorySize, constexpr_pow) {
    // check custom constexpr power function
    EXPECT_EQ(plssvm::detail::constexpr_pow(0, 1), 0);
    EXPECT_EQ(plssvm::detail::constexpr_pow(2, 0), 1);
    EXPECT_EQ(plssvm::detail::constexpr_pow(2, 3), 8);
    EXPECT_EQ(plssvm::detail::constexpr_pow(1024, 2), 1'048'576);
}

TEST(MemorySize, hash) {
    // hash a memory size
    const plssvm::detail::memory_size mem{ 1024ULL };
    const std::size_t hash_value = std::hash<plssvm::detail::memory_size>{}(mem);

    // hash the same value as unsigned long
    const std::size_t correct_hash_value = std::hash<unsigned long long>{}(mem.num_bytes());

    // check hash values
    EXPECT_EQ(hash_value, correct_hash_value);
}

//*************************************************************************************************************************************//
//                                                           custom literals                                                           //
//*************************************************************************************************************************************//
TEST(MemorySizeLiterals, base_ten_ull) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(0_B, plssvm::detail::memory_size{ 0ULL });
    EXPECT_EQ(8_B, plssvm::detail::memory_size{ 8ULL });
    EXPECT_EQ(1_KiB, plssvm::detail::memory_size{ 1024ULL });
    EXPECT_EQ(2_KiB, plssvm::detail::memory_size{ 2048ULL });
    EXPECT_EQ(1_MiB, plssvm::detail::memory_size{ 1'048'576ULL });
    EXPECT_EQ(3_MiB, plssvm::detail::memory_size{ 3'145'728ULL });
    EXPECT_EQ(1_GiB, plssvm::detail::memory_size{ 1'073'741'824ULL });
    EXPECT_EQ(4_GiB, plssvm::detail::memory_size{ 4'294'967'296ULL });
    EXPECT_EQ(1_TiB, plssvm::detail::memory_size{ 1'099'511'627'776ULL });
    EXPECT_EQ(5_TiB, plssvm::detail::memory_size{ 5'497'558'138'880ULL });
    EXPECT_EQ(1024_TiB, plssvm::detail::memory_size{ 1'125'899'906'842'624ULL });
}

TEST(MemorySizeLiterals, base_ten_l) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(0.0_B, plssvm::detail::memory_size{ 0ULL });
    EXPECT_EQ(8.0_B, plssvm::detail::memory_size{ 8ULL });
    EXPECT_EQ(1.0_KiB, plssvm::detail::memory_size{ 1024ULL });
    EXPECT_EQ(0.5_KiB, plssvm::detail::memory_size{ 512ULL });
    EXPECT_EQ(1_MiB, plssvm::detail::memory_size{ 1'048'576ULL });
    EXPECT_EQ(0.5_MiB, plssvm::detail::memory_size{ 524'288ULL });
    EXPECT_EQ(1_GiB, plssvm::detail::memory_size{ 1'073'741'824ULL });
    EXPECT_EQ(0.5_GiB, plssvm::detail::memory_size{ 536'870'912ULL });
    EXPECT_EQ(1_TiB, plssvm::detail::memory_size{ 1'099'511'627'776ULL });
    EXPECT_EQ(0.5_TiB, plssvm::detail::memory_size{ 549'755'813'888ULL });
    EXPECT_EQ(0.0009765625_TiB, plssvm::detail::memory_size{ 1'073'741'824ULL });
}

TEST(MemorySizeLiterals, base_two_ull) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(1_KB, plssvm::detail::memory_size{ 1000ULL });
    EXPECT_EQ(2_KB, plssvm::detail::memory_size{ 2000ULL });
    EXPECT_EQ(1_MB, plssvm::detail::memory_size{ 1'000'000ULL });
    EXPECT_EQ(3_MB, plssvm::detail::memory_size{ 3'000'000ULL });
    EXPECT_EQ(1_GB, plssvm::detail::memory_size{ 1'000'000'000ULL });
    EXPECT_EQ(4_GB, plssvm::detail::memory_size{ 4'000'000'000ULL });
    EXPECT_EQ(1_TB, plssvm::detail::memory_size{ 1'000'000'000'000ULL });
    EXPECT_EQ(5_TB, plssvm::detail::memory_size{ 5'000'000'000'000ULL });
    EXPECT_EQ(1024_TB, plssvm::detail::memory_size{ 1'024'000'000'000'000ULL });
}

TEST(MemorySizeLiterals, base_two_l) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(1.0_KB, plssvm::detail::memory_size{ 1000ULL });
    EXPECT_EQ(0.5_KB, plssvm::detail::memory_size{ 500ULL });
    EXPECT_EQ(1_MB, plssvm::detail::memory_size{ 1'000'000ULL });
    EXPECT_EQ(0.5_MB, plssvm::detail::memory_size{ 500'000ULL });
    EXPECT_EQ(1_GB, plssvm::detail::memory_size{ 1'000'000'000ULL });
    EXPECT_EQ(0.5_GB, plssvm::detail::memory_size{ 500'000'000ULL });
    EXPECT_EQ(1_TB, plssvm::detail::memory_size{ 1'000'000'000'000ULL });
    EXPECT_EQ(0.5_TB, plssvm::detail::memory_size{ 500'000'000'000ULL });
    EXPECT_EQ(0.0001_TB, plssvm::detail::memory_size{ 100'000'000ULL });
}

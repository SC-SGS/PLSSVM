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

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_CONVERSION_TO_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

TEST(MemorySize, default_construct) {
    // default construct a memory size object
    const plssvm::detail::memory_size ms{};

    // 0 bytes should be reported
    EXPECT_EQ(ms.num_bytes(), 0ULL);
}
TEST(MemorySize, construct_from_ull) {
    // construct a memory size object using an unsigned long long
    const plssvm::detail::memory_size ms{ 1024ULL };
    EXPECT_EQ(ms.num_bytes(), 1024ULL);
}

TEST(MemorySize, num_bytes) {
    // create memory size object
    const plssvm::detail::memory_size ms{ 512 };
    // check getter
    EXPECT_EQ(ms.num_bytes(), 512ULL);
}

TEST(MemorySize, operator_add) {
    // create two memory size objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };

    // check addition
    const plssvm::detail::memory_size ms = ms1 + ms2;
    EXPECT_EQ(ms.num_bytes(), 1536);
}
TEST(MemorySize, operator_subtract) {
    // create two memory size objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };

    // check subtraction
    const plssvm::detail::memory_size ms = ms1 - ms2;
    EXPECT_EQ(ms.num_bytes(), 512);
}
TEST(MemorySize, operator_scale_mul) {
    // create two memory size objects
    const plssvm::detail::memory_size ms1{ 1024 };

    // check scale via multiplication
    const plssvm::detail::memory_size ms_r1 = ms1 * 0.5L;
    EXPECT_EQ(ms_r1.num_bytes(), 512);
    const plssvm::detail::memory_size ms_r2 = 0.5L * ms1;
    EXPECT_EQ(ms_r2.num_bytes(), 512);
}
TEST(MemorySize, operator_scale_div) {
    // create two memory size objects
    const plssvm::detail::memory_size ms1{ 1024 };

    // check addition via division
    const plssvm::detail::memory_size ms = ms1 / 2.0L;
    EXPECT_EQ(ms.num_bytes(), 512);
}
TEST(MemorySize, operator_factor) {
    // create two memory size objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };

    // check addition via division
    const long double factor = ms1 / ms2;
    EXPECT_EQ(factor, 2.0L);
}

TEST(MemorySize, relational_equal) {
    // create memory objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };
    const plssvm::detail::memory_size ms3{ 512 };

    // check for equality
    EXPECT_FALSE(ms1 == ms2);
    EXPECT_TRUE(ms2 == ms3);
    EXPECT_FALSE(ms3 == ms1);
}
TEST(MemorySize, relational_inequal) {
    // create memory objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };
    const plssvm::detail::memory_size ms3{ 512 };

    // check for equality
    EXPECT_TRUE(ms1 != ms2);
    EXPECT_FALSE(ms2 != ms3);
    EXPECT_TRUE(ms3 != ms1);
}
TEST(MemorySize, relational_less) {
    // create memory objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };
    const plssvm::detail::memory_size ms3{ 512 };

    // check for equality
    EXPECT_FALSE(ms1 < ms2);
    EXPECT_FALSE(ms2 < ms3);
    EXPECT_TRUE(ms3 < ms1);
}
TEST(MemorySize, relational_greater) {
    // create memory objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };
    const plssvm::detail::memory_size ms3{ 512 };

    // check for equality
    EXPECT_TRUE(ms1 > ms2);
    EXPECT_FALSE(ms2 > ms3);
    EXPECT_FALSE(ms3 > ms1);
}

TEST(MemorySize, relational_less_or_equal) {
    // create memory objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };
    const plssvm::detail::memory_size ms3{ 512 };

    // check for equality
    EXPECT_FALSE(ms1 <= ms2);
    EXPECT_TRUE(ms2 <= ms3);
    EXPECT_TRUE(ms3 <= ms1);
}
TEST(MemorySize, relational_greater_or_equal) {
    // create memory objects
    const plssvm::detail::memory_size ms1{ 1024 };
    const plssvm::detail::memory_size ms2{ 512 };
    const plssvm::detail::memory_size ms3{ 512 };

    // check for equality
    EXPECT_TRUE(ms1 >= ms2);
    EXPECT_TRUE(ms2 >= ms3);
    EXPECT_FALSE(ms3 >= ms1);
}

TEST(MemorySize, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 0ULL }, "0 B");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 8ULL }, "8 B");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1024ULL }, "1.00 KiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 2048ULL }, "2.00 KiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1048576ULL }, "1.00 MiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 3145728ULL }, "3.00 MiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1073741824ULL }, "1.00 GiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 4294967296ULL }, "4.00 GiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1099511627776ULL }, "1.00 TiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 5497558138880ULL }, "5.00 TiB");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::memory_size{ 1125899906842624ULL }, "1024.00 TiB");
}

TEST(MemorySize, from_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_FROM_STRING("0 B", plssvm::detail::memory_size{ 0ULL });
    EXPECT_CONVERSION_FROM_STRING("8 B", plssvm::detail::memory_size{ 8ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00 KiB", plssvm::detail::memory_size{ 1024ULL });
    EXPECT_CONVERSION_FROM_STRING("0.5 KB", plssvm::detail::memory_size{ 500ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00 MiB", plssvm::detail::memory_size{ 1048576ULL });
    EXPECT_CONVERSION_FROM_STRING("0.4 MB", plssvm::detail::memory_size{ 400000ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00 GiB", plssvm::detail::memory_size{ 1073741824ULL });
    EXPECT_CONVERSION_FROM_STRING("0.3 GB", plssvm::detail::memory_size{ 300000000ULL });
    EXPECT_CONVERSION_FROM_STRING("1.00TiB", plssvm::detail::memory_size{ 1099511627776ULL });
    EXPECT_CONVERSION_FROM_STRING("0.2TB", plssvm::detail::memory_size{ 200000000000ULL });
}
TEST(MemorySize, from_string_unknown) {
    // foo isn't a valid backend_type
    std::istringstream input{ "0Bit" };
    plssvm::detail::memory_size ms{};
    input >> ms;
    EXPECT_TRUE(input.fail());
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
    EXPECT_EQ(1_MiB, plssvm::detail::memory_size{ 1048576ULL });
    EXPECT_EQ(3_MiB, plssvm::detail::memory_size{ 3145728ULL });
    EXPECT_EQ(1_GiB, plssvm::detail::memory_size{ 1073741824ULL });
    EXPECT_EQ(4_GiB, plssvm::detail::memory_size{ 4294967296ULL });
    EXPECT_EQ(1_TiB, plssvm::detail::memory_size{ 1099511627776ULL });
    EXPECT_EQ(5_TiB, plssvm::detail::memory_size{ 5497558138880ULL });
    EXPECT_EQ(1024_TiB, plssvm::detail::memory_size{ 1125899906842624ULL });
}
TEST(MemorySizeLiterals, base_ten_l) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(0.0_B, plssvm::detail::memory_size{ 0ULL });
    EXPECT_EQ(8.0_B, plssvm::detail::memory_size{ 8ULL });
    EXPECT_EQ(1.0_KiB, plssvm::detail::memory_size{ 1024ULL });
    EXPECT_EQ(0.5_KiB, plssvm::detail::memory_size{ 512ULL });
    EXPECT_EQ(1_MiB, plssvm::detail::memory_size{ 1048576ULL });
    EXPECT_EQ(0.5_MiB, plssvm::detail::memory_size{ 524288ULL });
    EXPECT_EQ(1_GiB, plssvm::detail::memory_size{ 1073741824ULL });
    EXPECT_EQ(0.5_GiB, plssvm::detail::memory_size{ 536870912ULL });
    EXPECT_EQ(1_TiB, plssvm::detail::memory_size{ 1099511627776ULL });
    EXPECT_EQ(0.5_TiB, plssvm::detail::memory_size{ 549755813888ULL });
    EXPECT_EQ(0.0009765625_TiB, plssvm::detail::memory_size{ 1073741824ULL });
}

TEST(MemorySizeLiterals, base_two_ull) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(1_KB, plssvm::detail::memory_size{ 1000ULL });
    EXPECT_EQ(2_KB, plssvm::detail::memory_size{ 2000ULL });
    EXPECT_EQ(1_MB, plssvm::detail::memory_size{ 1000000ULL });
    EXPECT_EQ(3_MB, plssvm::detail::memory_size{ 3000000ULL });
    EXPECT_EQ(1_GB, plssvm::detail::memory_size{ 1000000000ULL });
    EXPECT_EQ(4_GB, plssvm::detail::memory_size{ 4000000000ULL });
    EXPECT_EQ(1_TB, plssvm::detail::memory_size{ 1000000000000ULL });
    EXPECT_EQ(5_TB, plssvm::detail::memory_size{ 5000000000000ULL });
    EXPECT_EQ(1024_TB, plssvm::detail::memory_size{ 1024000000000000ULL });
}
TEST(MemorySizeLiterals, base_two_l) {
    // check if literals are correct
    using namespace plssvm::detail::literals;

    EXPECT_EQ(1.0_KB, plssvm::detail::memory_size{ 1000ULL });
    EXPECT_EQ(0.5_KB, plssvm::detail::memory_size{ 500ULL });
    EXPECT_EQ(1_MB, plssvm::detail::memory_size{ 1000000ULL });
    EXPECT_EQ(0.5_MB, plssvm::detail::memory_size{ 500000ULL });
    EXPECT_EQ(1_GB, plssvm::detail::memory_size{ 1000000000ULL });
    EXPECT_EQ(0.5_GB, plssvm::detail::memory_size{ 500000000ULL });
    EXPECT_EQ(1_TB, plssvm::detail::memory_size{ 1000000000000ULL });
    EXPECT_EQ(0.5_TB, plssvm::detail::memory_size{ 500000000000ULL });
    EXPECT_EQ(0.0001_TB, plssvm::detail::memory_size{ 100000000ULL });
}
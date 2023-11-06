/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the utility header.
 */

#include "plssvm/detail/utility.hpp"

#include "plssvm/default_value.hpp"  // plssvm::default_value

#include "naming.hpp"         // naming::{test_parameter_to_name}
#include "types_to_test.hpp"  // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, test_parameter_type_at_t}

#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ::testing::Test

#include <map>            // std::map
#include <regex>          // std::regex, std::regex::extended, std::regex_match
#include <set>            // std::set
#include <string>         // std::string
#include <tuple>          // std::tuple
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector

TEST(Utility, plssvm_is_defined_macro) {
    // the following macro is ALWAYS defined in PLSSVM
    constexpr bool is_defined = PLSSVM_IS_DEFINED(PLSSVM_BUILD_TYPE);
    EXPECT_TRUE(is_defined);

    // the following macro will never be defined in PLSSVM
    constexpr bool not_defined = PLSSVM_IS_DEFINED(PLSSVM_THIS_MACRO_WILL_NEVER_BE_DEFINED);
    EXPECT_FALSE(not_defined);
}

TEST(Utility, get) {
    EXPECT_EQ(plssvm::detail::get<0>(0, 1, 2, 3, 4), 0);
    EXPECT_EQ(plssvm::detail::get<1>(0, 1.5, 2, 3, 4), 1.5);
    EXPECT_EQ(plssvm::detail::get<2>(0, 1, -2, 3, 4), -2);
    EXPECT_EQ(plssvm::detail::get<3>(0, 1, 2, 'a', 4), 'a');
    EXPECT_EQ(plssvm::detail::get<4>(0, 1, 2, 3, std::string{ "abc" }), std::string{ "abc" });
}

TEST(Utility, to_underlying_int) {
    // clang-format off
    enum class int_enum { a, b, c = 10 };
    // clang-format on

    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::a), 0);
    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::b), 1);
    EXPECT_EQ(plssvm::detail::to_underlying(int_enum::c), 10);
}

TEST(Utility, to_underlying_char) {
    // clang-format off
    enum class char_enum : char { a = 'a', b = 'b', c = 'c' };
    // clang-format on

    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::a), 'a');
    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::b), 'b');
    EXPECT_EQ(plssvm::detail::to_underlying(char_enum::c), 'c');
}

TEST(Utility, to_underlying_default_value_int) {
    // clang-format off
    enum class int_enum { a, b, c = 10 };
    // clang-format on

    const plssvm::default_value<int_enum> int_default_a{ plssvm::default_init{ int_enum::a } };
    EXPECT_EQ(plssvm::detail::to_underlying(int_default_a), 0);
    const plssvm::default_value<int_enum> int_default_b{ plssvm::default_init{ int_enum::b } };
    EXPECT_EQ(plssvm::detail::to_underlying(int_default_b), 1);
    const plssvm::default_value<int_enum> int_default_c{ plssvm::default_init{ int_enum::c } };
    EXPECT_EQ(plssvm::detail::to_underlying(int_default_c), 10);
}

TEST(Utility, to_underlying_default_value_char) {
    // clang-format off
    enum class char_enum : char { a = 'a', b = 'b', c = 'c' };
    // clang-format on

    const plssvm::default_value<char_enum> char_default_a{ plssvm::default_init{ char_enum::a } };
    EXPECT_EQ(plssvm::detail::to_underlying(char_default_a), 'a');
    const plssvm::default_value<char_enum> char_default_b{ plssvm::default_init{ char_enum::b } };
    EXPECT_EQ(plssvm::detail::to_underlying(char_default_b), 'b');
    const plssvm::default_value<char_enum> char_default_c{ plssvm::default_init{ char_enum::c } };
    EXPECT_EQ(plssvm::detail::to_underlying(char_default_c), 'c');
}

// the map container types to test
using map_types = std::tuple<std::map<int, int>, std::unordered_map<int, int>>;
using map_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<map_types>>;

// test fixture for map like classes
template <typename T>
class UtilityMapContainer : public ::testing::Test {
  public:
    /// The type of the encapsulated map.
    using map_type = util::test_parameter_type_at_t<0, T>;

  protected:
    /**
     * @brief Initialize the map.
     */
    void SetUp() override {
        // initialize map
        map_ = { { 0, 0 }, { 1, 1 } };
    }
    /**
     * @brief Return the encapsulated map.
     * @return the map (`[[nodiscard]]`)
     */
    map_type &get_map() { return map_; }

  private:
    map_type map_;
};
TYPED_TEST_SUITE(UtilityMapContainer, map_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(UtilityMapContainer, erase_if) {
    EXPECT_EQ(plssvm::detail::erase_if(this->get_map(), [](const typename TestFixture::map_type::value_type value) { return value.second % 2 == 0; }), 1);
    EXPECT_EQ(this->get_map().size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->get_map(), [](const typename TestFixture::map_type::value_type value) { return value.second % 2 == 0; }), 0);
    EXPECT_EQ(this->get_map().size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->get_map(), [](const typename TestFixture::map_type::value_type value) { return value.second % 2 == 1; }), 1);
    EXPECT_TRUE(this->get_map().empty());
}
TYPED_TEST(UtilityMapContainer, contains) {
    EXPECT_TRUE(plssvm::detail::contains(this->get_map(), 0));
    EXPECT_TRUE(plssvm::detail::contains(this->get_map(), 1));
    EXPECT_FALSE(plssvm::detail::contains(this->get_map(), 2));
    EXPECT_FALSE(plssvm::detail::contains(this->get_map(), -1));
}

// the set container types to test
using set_types = std::tuple<std::set<int>, std::unordered_set<int>>;
using set_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<set_types>>;

// test fixture for set like classes
template <typename T>
class UtilitySetContainer : public ::testing::Test {
  public:
    /// The type of the encapsulated set.
    using set_type = util::test_parameter_type_at_t<0, T>;

  protected:
    /**
     * @brief Initialize the set.
     */
    void SetUp() override {
        // initialize set
        set_ = { 0, 1 };
    }
    /**
     * @brief Return the encapsulated set.
     * @return the set (`[[nodiscard]]`)
     */
    set_type &get_set() { return set_; }

  private:
    set_type set_;
};
TYPED_TEST_SUITE(UtilitySetContainer, set_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(UtilitySetContainer, erase_if) {
    EXPECT_EQ(plssvm::detail::erase_if(this->get_set(), [](const typename TestFixture::set_type::value_type value) { return value % 2 == 0; }), 1);
    EXPECT_EQ(this->get_set().size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->get_set(), [](const typename TestFixture::set_type::value_type value) { return value % 2 == 0; }), 0);
    EXPECT_EQ(this->get_set().size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->get_set(), [](const typename TestFixture::set_type::value_type value) { return value % 2 == 1; }), 1);
    EXPECT_TRUE(this->get_set().empty());
}
TYPED_TEST(UtilitySetContainer, contains) {
    EXPECT_TRUE(plssvm::detail::contains(this->get_set(), 0));
    EXPECT_TRUE(plssvm::detail::contains(this->get_set(), 1));
    EXPECT_FALSE(plssvm::detail::contains(this->get_set(), 2));
    EXPECT_FALSE(plssvm::detail::contains(this->get_set(), -1));
}

// the vector container types to test
using vector_types = std::tuple<std::vector<int>>;
using vector_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<vector_types>>;

// test fixture for vector like classes
template <typename T>
class UtilityVectorContainer : public ::testing::Test {
  public:
    /// The type of the encapsulated vector.
    using vector_type = util::test_parameter_type_at_t<0, T>;

  protected:
    /**
     * @brief Initialize the vector.
     */
    void SetUp() override {
        // initialize vector
        vec_ = { 0, 1 };
    }
    /**
     * @brief Return the encapsulated vector.
     * @return the set (`[[nodiscard]]`)
     */
    vector_type &get_vector() { return vec_; }

  private:
    vector_type vec_;
};

TYPED_TEST_SUITE(UtilityVectorContainer, vector_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(UtilityVectorContainer, erase_if) {
    EXPECT_EQ(plssvm::detail::erase_if(this->get_vector(), [](const typename TestFixture::vector_type::value_type value) { return value % 2 == 0; }), 1);
    EXPECT_EQ(this->get_vector().size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->get_vector(), [](const typename TestFixture::vector_type::value_type value) { return value % 2 == 0; }), 0);
    EXPECT_EQ(this->get_vector().size(), 1);
    EXPECT_EQ(plssvm::detail::erase_if(this->get_vector(), [](const typename TestFixture::vector_type::value_type value) { return value % 2 == 1; }), 1);
    EXPECT_TRUE(this->get_vector().empty());
}
TYPED_TEST(UtilityVectorContainer, contains) {
    EXPECT_TRUE(plssvm::detail::contains(this->get_vector(), 0));
    EXPECT_TRUE(plssvm::detail::contains(this->get_vector(), 1));
    EXPECT_FALSE(plssvm::detail::contains(this->get_vector(), 2));
    EXPECT_FALSE(plssvm::detail::contains(this->get_vector(), -1));
}

TEST(Utility, current_date_time) {
    // test if the current date time matches the pattern
    EXPECT_TRUE(std::regex_match(std::string{ plssvm::detail::current_date_time() },
                                 std::regex{ "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}", std::regex::extended }));
}

TEST(Utility, get_system_memory) {
    // the available system memory must be greater than 0!
    EXPECT_GT(plssvm::detail::get_system_memory().num_bytes(), 0ULL);
}
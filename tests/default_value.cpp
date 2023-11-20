/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the default_value wrapper.
 */

#include "plssvm/default_value.hpp"

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::[label_type_gtest, test_parameter_type_at_t}

#include "gtest/gtest.h"  // TEST, TYPED_TEST, TEST_P, TYPED_TEST_SUITE, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DOUBLE_EQ
                          // ::testing::{Test, WithParamInterface, Values}

#include <functional>   // std::hash
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple, std::get
#include <utility>      // std::move, std::swap
#include <vector>       // std::vector

//*************************************************************************************************************************************//
//                                                            default_init                                                             //
//*************************************************************************************************************************************//

template <typename T>
class DefaultInitDefault : public ::testing::Test {};
TYPED_TEST_SUITE(DefaultInitDefault, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DefaultInitDefault, default_construct) {
    using type = util::test_parameter_type_at_t<0, TypeParam>;

    // check for correct default construction
    EXPECT_EQ(plssvm::default_init<type>{}.value, type{});
}

class DefaultInitExplicit : public ::testing::Test {};
class DefaultInitIntegral : public DefaultInitExplicit, public ::testing::WithParamInterface<int> {};
TEST_P(DefaultInitIntegral, explicit_construct) {
    const auto val = GetParam();

    // check for correct construction
    EXPECT_EQ(plssvm::default_init{ val }.value, val);
}
INSTANTIATE_TEST_SUITE_P(DefaultInitExplicit, DefaultInitIntegral, ::testing::Values(0, 1, 2, 3, 42, -1, -5), naming::pretty_print_escaped_string<DefaultInitIntegral>);

class DefaultInitFloatingPoint : public DefaultInitExplicit, public ::testing::WithParamInterface<double> {};
TEST_P(DefaultInitFloatingPoint, explicit_construct) {
    const auto val = GetParam();

    // check for correct construction
    EXPECT_EQ(plssvm::default_init{ val }.value, val);
}
INSTANTIATE_TEST_SUITE_P(DefaultInitExplicit, DefaultInitFloatingPoint, ::testing::Values(0.0, 1.2, 2.5, 3.38748, 42.1, -1, -5.22), naming::pretty_print_escaped_string<DefaultInitFloatingPoint>);

class DefaultInitString : public DefaultInitExplicit, public ::testing::WithParamInterface<std::string> {};
TEST_P(DefaultInitString, explicit_construct) {
    const auto val = GetParam();

    // check for correct construction
    EXPECT_EQ(plssvm::default_init{ val }.value, val);
}
INSTANTIATE_TEST_SUITE_P(DefaultInitExplicit, DefaultInitString, ::testing::Values("", "foo", "bar", "baz", "Hello World"), naming::pretty_print_escaped_string<DefaultInitString>);

//*************************************************************************************************************************************//
//                                                            default_value                                                            //
//*************************************************************************************************************************************//

TEST(DefaultValue, default_init) {
    // create default_value
    const plssvm::default_value val{ plssvm::default_init{ 42 } };

    // a default value has been assigned
    EXPECT_TRUE(val.is_default());
    EXPECT_EQ(val.value(), 42);
    EXPECT_EQ(val.get_default(), 42);
}

TEST(DefaultValue, assign_non_default) {
    // create default_value
    plssvm::default_value<double> val{};

    // must be default
    EXPECT_TRUE(val.is_default());

    // assign non-default value
    val = 3.1415;
    // value must now be a non-default value
    EXPECT_FALSE(val.is_default());
    EXPECT_DOUBLE_EQ(val.value(), 3.1415);
    EXPECT_DOUBLE_EQ(val.get_default(), 0.0);
}

TEST(DefaultValue, copy_construct_default) {
    // create first default_value
    const plssvm::default_value val1{ plssvm::default_init{ 3.1415 } };

    // copy-construct second default value
    const plssvm::default_value<int> val2{ val1 };

    // check for correct values
    EXPECT_TRUE(val2.is_default());
    EXPECT_EQ(val2.value(), 3);
    EXPECT_EQ(val2.get_default(), 3);
    // val1 must not have changed
    EXPECT_TRUE(val1.is_default());
    EXPECT_EQ(val1.value(), 3.1415);
    EXPECT_EQ(val1.get_default(), 3.1415);
}
TEST(DefaultValue, copy_construct_non_default) {
    // create first default_value
    plssvm::default_value<double> val1{};
    val1 = 3.1415;

    // copy-construct second default value
    const plssvm::default_value<int> val2{ val1 };

    // check for correct values
    EXPECT_FALSE(val2.is_default());
    EXPECT_EQ(val2.value(), 3);
    EXPECT_EQ(val2.get_default(), 0);
    // val1 must not have changed
    EXPECT_FALSE(val1.is_default());
    EXPECT_DOUBLE_EQ(val1.value(), 3.1415);
    EXPECT_DOUBLE_EQ(val1.get_default(), 0.0);
}

TEST(DefaultValue, move_construct_default) {
    // create first default_value
    plssvm::default_value val1{ plssvm::default_init<std::string>{ "Hello, World!" } };

    // copy-construct second default value
    const plssvm::default_value<std::string> val2{ std::move(val1) };

    // check for correct values
    EXPECT_TRUE(val2.is_default());
    EXPECT_EQ(val2.value(), "Hello, World!");
    EXPECT_EQ(val2.get_default(), "Hello, World!");
}
TEST(DefaultValue, move_construct_non_default) {
    // create first default_value
    plssvm::default_value<std::string> val1{};
    val1 = "foo bar baz";

    // copy-construct second default value
    const plssvm::default_value<std::string> val2{ std::move(val1) };

    // check for correct values
    EXPECT_FALSE(val2.is_default());
    EXPECT_EQ(val2.value(), "foo bar baz");
    EXPECT_EQ(val2.get_default(), "");
}

TEST(DefaultValue, copy_assign_default) {
    // create two default_values
    const plssvm::default_value val1{ plssvm::default_init{ 3.1415 } };
    plssvm::default_value val2{ plssvm::default_init{ 42 } };

    // copy-assign second default value
    val2 = val1;

    // check for correct values
    EXPECT_TRUE(val2.is_default());
    EXPECT_EQ(val2.value(), 3);
    EXPECT_EQ(val2.get_default(), 3);
    // val1 must not have changed!
    EXPECT_TRUE(val1.is_default());
    EXPECT_DOUBLE_EQ(val1.value(), 3.1415);
    EXPECT_DOUBLE_EQ(val1.get_default(), 3.1415);
}
TEST(DefaultValue, copy_assign_non_default) {
    // create two default_values
    plssvm::default_value val1{ plssvm::default_init{ 3.1415 } };
    val1 = 2.7182;
    plssvm::default_value val2{ plssvm::default_init{ 42 } };

    // copy-assign second default value
    val2 = val1;

    // check for correct values
    EXPECT_FALSE(val2.is_default());
    EXPECT_EQ(val2.value(), 2);
    EXPECT_EQ(val2.get_default(), 3);
    // val1 must not have changed!
    EXPECT_FALSE(val1.is_default());
    EXPECT_DOUBLE_EQ(val1.value(), 2.7182);
    EXPECT_DOUBLE_EQ(val1.get_default(), 3.1415);
}

TEST(DefaultValue, move_assign_default) {
    // create two default_values
    plssvm::default_value val1{ plssvm::default_init<std::string>{ "AAA" } };
    plssvm::default_value val2{ plssvm::default_init<std::string>{ "BBB" } };

    // copy-assign second default value
    val2 = std::move(val1);

    // check for correct values
    EXPECT_TRUE(val2.is_default());
    EXPECT_EQ(val2.value(), "AAA");
    EXPECT_EQ(val2.get_default(), "AAA");
}
TEST(DefaultValue, move_assign_non_default) {
    // create two default_values
    plssvm::default_value val1{ plssvm::default_init<std::string>{ "AAA" } };
    val1 = "CCC";
    plssvm::default_value val2{ plssvm::default_init<std::string>{ "BBB" } };

    // copy-assign second default value
    val2 = val1;

    // check for correct values
    EXPECT_FALSE(val2.is_default());
    EXPECT_EQ(val2.value(), "CCC");
    EXPECT_EQ(val2.get_default(), "AAA");
}

TEST(DefaultValue, value_default) {
    // create default_value
    const plssvm::default_value val{ plssvm::default_init{ 42 } };

    // check for correct value
    EXPECT_EQ(val.value(), 42);
}
TEST(DefaultValue, value_non_default) {
    // create default_value
    plssvm::default_value val{ plssvm::default_init<std::string>{ "AAA" } };
    val = "BBB";

    // check for correct values
    EXPECT_EQ(val.value(), "BBB");
}

TEST(DefaultValue, implicit_conversion_default) {
    // create default_value
    const plssvm::default_value val{ plssvm::default_init{ 42 } };

    // check for correct value
    EXPECT_EQ(static_cast<int>(val), 42);
}
TEST(DefaultValue, implicit_conversion_non_default) {
    // create default_value
    plssvm::default_value val{ plssvm::default_init<std::string>{ "AAA" } };
    val = "BBB";

    // check for correct values
    EXPECT_EQ(static_cast<std::string>(val), "BBB");
}

TEST(DefaultValue, get_default_default) {
    // create default_value
    const plssvm::default_value val{ plssvm::default_init{ 42 } };

    // default_init -> must be default
    EXPECT_EQ(val.get_default(), 42);
}
TEST(DefaultValue, get_default_non_default) {
    // create default_value
    plssvm::default_value val{ plssvm::default_init<std::string>{ "Hello, World!" } };
    val = "foo bar baz";

    // default value overwritten -> must not be default
    EXPECT_EQ(val.get_default(), "Hello, World!");
}

TEST(DefaultValue, is_default_default) {
    // create default_value
    const plssvm::default_value val{ plssvm::default_init{ 42 } };

    // default_init -> must be default
    EXPECT_TRUE(val.is_default());
}
TEST(DefaultValue, is_default_non_default) {
    // create default_value
    plssvm::default_value val{ plssvm::default_init{ "Hello, World!" } };
    val = "foo bar baz";

    // default value overwritten -> must not be default
    EXPECT_FALSE(val.is_default());
}

TEST(DefaultValue, swap_member_function) {
    // create two default_values
    plssvm::default_value val1{ plssvm::default_init{ 1 } };
    plssvm::default_value val2{ plssvm::default_init{ 2 } };
    val2 = 3;

    // check values before swap
    EXPECT_TRUE(val1.is_default());
    EXPECT_EQ(val1.value(), 1);
    EXPECT_EQ(val1.get_default(), 1);
    EXPECT_FALSE(val2.is_default());
    EXPECT_EQ(val2.value(), 3);
    EXPECT_EQ(val2.get_default(), 2);

    // swap contents
    val1.swap(val2);

    // check if contents were correctly swapped
    EXPECT_FALSE(val1.is_default());
    EXPECT_EQ(val1.value(), 3);
    EXPECT_EQ(val1.get_default(), 2);
    EXPECT_TRUE(val2.is_default());
    EXPECT_EQ(val2.value(), 1);
    EXPECT_EQ(val2.get_default(), 1);
}

TEST(DefaultValue, reset_default) {
    // create default_value
    plssvm::default_value val{ plssvm::default_init{ 42 } };

    // reset value
    val.reset();

    // check values
    EXPECT_TRUE(val.is_default());
    EXPECT_EQ(val.value(), 42);
    EXPECT_EQ(val.get_default(), 42);
}
TEST(DefaultValue, reset_non_default) {
    // create default_value
    plssvm::default_value val{ plssvm::default_init{ 42 } };
    val = 64;

    // reset value
    val.reset();

    // check values
    EXPECT_TRUE(val.is_default());
    EXPECT_EQ(val.value(), 42);
    EXPECT_EQ(val.get_default(), 42);
}

TEST(DefaultValue, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::default_value{ plssvm::default_init{ 1 } }, "1");
    EXPECT_CONVERSION_TO_STRING(plssvm::default_value{ plssvm::default_init{ 3.1415 } }, "3.1415");
    EXPECT_CONVERSION_TO_STRING(plssvm::default_value{ plssvm::default_init{ -4 } }, "-4");
    EXPECT_CONVERSION_TO_STRING(plssvm::default_value{ plssvm::default_init{ "Hello World" } }, "Hello World");
}
TEST(DefaultValue, from_string) {
    // check conversion from std::string
    plssvm::default_value<int> val1{};
    val1 = 1;
    EXPECT_CONVERSION_FROM_STRING("1", val1);
    EXPECT_FALSE(val1.is_default());
    EXPECT_EQ(val1.get_default(), 0);

    plssvm::default_value<double> val2{};
    val2 = 3.1415;
    EXPECT_CONVERSION_FROM_STRING("3.1415", val2);
    EXPECT_FALSE(val2.is_default());
    EXPECT_DOUBLE_EQ(val2.get_default(), 0.0);

    plssvm::default_value<int> val3{ plssvm::default_init{ 42 } };
    val3 = -4;
    EXPECT_CONVERSION_FROM_STRING("-4", val3);
    EXPECT_FALSE(val3.is_default());
    EXPECT_EQ(val3.get_default(), 42);

    plssvm::default_value<std::string> val4{};
    val4 = "foo";
    EXPECT_CONVERSION_FROM_STRING("foo", val4);
    EXPECT_FALSE(val4.is_default());
    EXPECT_EQ(val4.get_default(), std::string{ "" });
}

TEST(DefaultValue, swap_free_function) {
    // create two default_values
    plssvm::default_value val1{ plssvm::default_init{ 1 } };
    plssvm::default_value val2{ plssvm::default_init{ 2 } };
    val2 = 3;

    // check values before swap
    EXPECT_TRUE(val1.is_default());
    EXPECT_EQ(val1.value(), 1);
    EXPECT_EQ(val1.get_default(), 1);
    EXPECT_FALSE(val2.is_default());
    EXPECT_EQ(val2.value(), 3);
    EXPECT_EQ(val2.get_default(), 2);

    // swap contents
    std::swap(val1, val2);

    // check if contents were correctly swapped
    EXPECT_FALSE(val1.is_default());
    EXPECT_EQ(val1.value(), 3);
    EXPECT_EQ(val1.get_default(), 2);
    EXPECT_TRUE(val2.is_default());
    EXPECT_EQ(val2.value(), 1);
    EXPECT_EQ(val2.get_default(), 1);
}

using relation_op_func_ptr = bool (*)(const plssvm::default_value<int> &, const plssvm::default_value<int> &);
class DefaultValueRelational : public ::testing::TestWithParam<std::tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>> {
  protected:
    void SetUp() override {
        val1_ = 42;
    }

    /**
     * @brief Return the first default_value used for the relational operator overloads tests.
     * @return the default value (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::default_value<int> &get_val1() const noexcept { return val1_; }
    /**
     * @brief Return the second default_value used for the relational operator overloads tests.
     * @return the default value (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::default_value<int> &get_val2() const noexcept { return val2_; }
    /**
     * @brief Return the third default_value used for the relational operator overloads tests.
     * @return the default value (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::default_value<int> &get_val3() const noexcept { return val3_; }

  private:
    /// A default value used to test the relational operator overloads.
    plssvm::default_value<int> val1_{ plssvm::default_init{ 1 } };
    /// A default value used to test the relational operator overloads.
    plssvm::default_value<int> val2_{ plssvm::default_init{ 1 } };
    /// A default value used to test the relational operator overloads.
    plssvm::default_value<int> val3_{ plssvm::default_init{ 42 } };
};

TEST_P(DefaultValueRelational, relational_operators) {
    auto [op, op_name, booleans] = GetParam();

    // perform relational tests
    EXPECT_EQ(op(this->get_val1(), this->get_val2()), booleans[0]);
    EXPECT_EQ(op(this->get_val1(), this->get_val3()), booleans[1]);
    EXPECT_EQ(op(this->get_val2(), this->get_val3()), booleans[2]);

    // perform tests for idempotence
    EXPECT_EQ(op(this->get_val1(), this->get_val1()), booleans[3]);
    EXPECT_EQ(op(this->get_val2(), this->get_val2()), booleans[4]);
    EXPECT_EQ(op(this->get_val3(), this->get_val3()), booleans[5]);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(DefaultValue, DefaultValueRelational, ::testing::Values(
                std::make_tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>(&plssvm::operator==, "Equal", { false, true, false, true, true, true }),
                std::make_tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>(&plssvm::operator!=, "Unequal", { true, false, true, false, false, false }),
                std::make_tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>(&plssvm::operator<, "Less", { false, false, true, false, false, false }),
                std::make_tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>(&plssvm::operator>, "Greater", { true, false, false, false, false, false }),
                std::make_tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>(&plssvm::operator<=, "LessOrEqual", { false, true, true, true, true, true }),
                std::make_tuple<relation_op_func_ptr, std::string_view, std::vector<bool>>(&plssvm::operator>=, "GreaterOrEqual", { true, true, false, true, true, true })),
                naming::pretty_print_default_value_relational<DefaultValueRelational>);
// clang-format on

TEST(DefaultValue, is_default_value_trait) {
    // create a default_value
    const plssvm::default_value val{ plssvm::default_init{ 42 } };

    // test if the type_trait is correct
    EXPECT_TRUE(plssvm::is_default_value<decltype(val)>::value);
    EXPECT_TRUE(plssvm::is_default_value_v<decltype(val)>);
    EXPECT_FALSE(plssvm::is_default_value<typename decltype(val)::value_type>::value);
    EXPECT_FALSE(plssvm::is_default_value_v<typename decltype(val)::value_type>);
}

TEST(DefaultValue, hash) {
    // create default_value and hash it
    const plssvm::default_value val1{ plssvm::default_init{ 42 } };
    EXPECT_EQ(std::hash<plssvm::default_value<int>>{}(val1), std::hash<int>{}(42));

    const plssvm::default_value val2{ plssvm::default_init<std::string>{ "Hello, World!" } };
    EXPECT_EQ(std::hash<plssvm::default_value<std::string>>{}(val2), std::hash<std::string>{}("Hello, World!"));

    plssvm::default_value val3{ plssvm::default_init{ 3.1415 } };
    val3 = 2.7182;
    EXPECT_EQ(std::hash<plssvm::default_value<double>>{}(val3), std::hash<double>{}(2.7182));
}
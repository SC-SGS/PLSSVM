/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for simple_any implementation.
 */

#include "plssvm/detail/move_only_any.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <algorithm>  // std::swap
#include <memory>     // std::shared_ptr, std::make_shared
#include <string>     // std::string
#include <tuple>      // std::ignore
#include <utility>    // std::in_place_type, std::move
#include <vector>     // std::vector

TEST(BadMoveOnlyCastException, exception) {
    EXPECT_THROW_WHAT(throw plssvm::detail::bad_move_only_any_cast{}, plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, default_construct) {
    // default construct a move_only_any
    const plssvm::detail::move_only_any a{};

    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());
}

TEST(MoveOnlyAny, construct) {
    // construct move_only_any objects
    const plssvm::detail::move_only_any a1{ 42 };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a1), 42);

    const plssvm::detail::move_only_any a2{ 3.1415 };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<double>(a2), 3.1415);

    const plssvm::detail::move_only_any a3{ 'a' };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<char>(a3), 'a');

    const plssvm::detail::move_only_any a4{ true };
    EXPECT_TRUE(plssvm::detail::move_only_any_cast<bool>(a4));

    const plssvm::detail::move_only_any a5{ std::string{ "Test" } };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a5), std::string{ "Test" });

    const plssvm::detail::move_only_any a6{ std::vector<int>{ 1, 2, 3 } };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a6), (std::vector<int>{ 1, 2, 3 }));

    const plssvm::detail::move_only_any a7{ std::make_shared<float>(1.0f) };
    EXPECT_EQ(*plssvm::detail::move_only_any_cast<std::shared_ptr<float>>(a7), 1.0f);
}

TEST(MoveOnlyAny, construct_in_place) {
    // construct a move_only_any object using std::in_place_type
    const plssvm::detail::move_only_any a{ std::in_place_type<std::string>, std::string::size_type{ 10 }, 'a' };

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{ "aaaaaaaaaa" });
}

TEST(MoveOnlyAny, construct_in_place_with_initializer_list) {
    // construct a move_only_any object using std::in_place_type
    const plssvm::detail::move_only_any a{ std::in_place_type<std::vector<int>>, { 0, 1, 2, 3 } };

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a), (std::vector<int>{ 0, 1, 2, 3 }));
}

TEST(MoveOnlyAny, assignment_operator) {
    // default construct a move_only_any
    plssvm::detail::move_only_any a{};
    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());

    // assign new object to the move_only_any
    a = std::string{ "Hello, World!" };

    // check contained value
    EXPECT_TRUE(a.has_value());
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), (std::string{ "Hello, World!" }));
}

TEST(MoveOnlyAny, emplace) {
    // default construct a move_only_any
    plssvm::detail::move_only_any a{};
    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());

    // emplace new object in the move_only_any
    a.emplace<std::string>(10, 'b');

    // the move_only_any object must contain a string containing ten 'b' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{ "bbbbbbbbbb" });
}

TEST(MoveOnlyAny, emplace_with_initializer_list) {
    // default construct a move_only_any
    plssvm::detail::move_only_any a{};
    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());

    // emplace new object in the move_only_any
    a.emplace<std::vector<int>>({ 0, 1, 2, 3 });

    // the move_only_any object must contain a string containing ten 'b' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a), (std::vector<int>{ 0, 1, 2, 3 }));
}

TEST(MoveOnlyAny, reset) {
    // create move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // the constructed any should contain an object
    EXPECT_TRUE(a.has_value());

    // reset the move_only_any
    a.reset();

    // now, the move_only_any object should not contain an object anymore
    EXPECT_FALSE(a.has_value());
}

TEST(MoveOnlyAny, swap_member_function) {
    // create two move_only_any objects
    plssvm::detail::move_only_any a1{ 42 };
    plssvm::detail::move_only_any a2{ 3.1415 };

    // swap both any objects
    a1.swap(a2);

    // check whether the content changed
    EXPECT_EQ(plssvm::detail::move_only_any_cast<double>(a1), 3.1415);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a2), 42);
}

TEST(MoveOnlyAny, has_value) {
    // create move_only_any object that should contain an object
    plssvm::detail::move_only_any a1{ 42 };
    EXPECT_TRUE(a1.has_value());

    // default constructed move_only_any should not contain an object
    plssvm::detail::move_only_any a2{};
    EXPECT_FALSE(a2.has_value());
}

TEST(MoveOnlyAny, type) {
    // default constructed move_only_any should return the typeid(void) on a call to .type()
    const plssvm::detail::move_only_any a1{};
    EXPECT_EQ(a1.type(), typeid(void));

    // normal constructed move_only_any should return the typeid of the contained type on a call to .type()
    const plssvm::detail::move_only_any a2{ 42 };
    EXPECT_EQ(a2.type(), typeid(int));
}

TEST(MoveOnlyAny, swap_free_function) {
    // create two move_only_any objects
    plssvm::detail::move_only_any a1{ 42 };
    plssvm::detail::move_only_any a2{ 3.1415 };

    // swap both any objects
    using std::swap;
    swap(a1, a2);

    // check whether the content changed
    EXPECT_EQ(plssvm::detail::move_only_any_cast<double>(a1), 3.1415);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a2), 42);
}

TEST(MoveOnlyAny, cast_const_lvalue_reference) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a), 42);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const int &>(a), 42);
}

TEST(MoveOnlyAny, cast_const_lvalue_reference_wrong_type) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<const float &>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, cast_lvalue_reference) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a), 42);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const int &>(a), 42);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int &>(a), 42);
}

TEST(MoveOnlyAny, cast_lvalue_reference_wrong_type) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<const float &>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float &>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, cast_rvalue_reference) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ std::string{ "Hello, World!" } };
    // retrieve the contained value and check for correctness
    auto str = plssvm::detail::move_only_any_cast<std::string &&>(std::move(a));
    EXPECT_EQ(str, (std::string{ "Hello, World!" }));
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{});
}

TEST(MoveOnlyAny, cast_rvalue_reference_wrong_type) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float &&>(std::move(a)), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, cast_const_pointer) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    const auto *ptr = plssvm::detail::move_only_any_cast<const int>(&a);
    EXPECT_EQ(*ptr, 42);
}

TEST(MoveOnlyAny, cast_const_pointer_wrong_type) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const float>(&a), nullptr);
}

TEST(MoveOnlyAny, cast_pointer) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    EXPECT_EQ(*plssvm::detail::move_only_any_cast<int>(&a), 42);
    EXPECT_EQ(*plssvm::detail::move_only_any_cast<const int>(&a), 42);
}

TEST(MoveOnlyAny, cast_pointer_wrong_type) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_EQ(plssvm::detail::move_only_any_cast<float>(&a), nullptr);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const float>(&a), nullptr);
}

TEST(MoveOnlyAny, make_move_only_any) {
    // construct a move_only_any object
    const auto a = plssvm::detail::make_move_only_any<std::string>(10, 'a');

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{ "aaaaaaaaaa" });
}

TEST(MoveOnlyAny, make_move_only_any_with_initializer_list) {
    // construct a move_only_any object
    const auto a = plssvm::detail::make_move_only_any<std::vector<int>>({ 0, 1, 2, 3 });

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a), (std::vector<int>{ 0, 1, 2, 3 }));
}

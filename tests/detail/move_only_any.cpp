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

TEST(BadMoveOnlyCastException, Exception) {
    const auto dummy = []() { throw plssvm::detail::bad_move_only_any_cast{}; };
    EXPECT_THROW_WHAT(dummy(), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, DefaultConstruct) {
    // default construct a move_only_any
    const plssvm::detail::move_only_any a{};

    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());
}

TEST(MoveOnlyAny, Construct) {
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

TEST(MoveOnlyAny, ConstructInPlace) {
    // construct a move_only_any object using std::in_place_type
    const plssvm::detail::move_only_any a{ std::in_place_type<std::string>, std::string::size_type{ 10 }, 'a' };

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{ "aaaaaaaaaa" });
}

TEST(MoveOnlyAny, ConstructInPlaceWithInitializerList) {
    // construct a move_only_any object using std::in_place_type
    const plssvm::detail::move_only_any a{ std::in_place_type<std::vector<int>>, { 0, 1, 2, 3 } };

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a), (std::vector<int>{ 0, 1, 2, 3 }));
}

TEST(MoveOnlyAny, AssignmentOperator) {
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

TEST(MoveOnlyAny, Emplace) {
    // default construct a move_only_any
    plssvm::detail::move_only_any a{};
    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());

    // emplace new object in the move_only_any
    a.emplace<std::string>(10, 'b');

    // the move_only_any object must contain a string containing ten 'b' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{ "bbbbbbbbbb" });
}

TEST(MoveOnlyAny, EmplaceWithInitializerList) {
    // default construct a move_only_any
    plssvm::detail::move_only_any a{};
    // the constructed any should not contain an object
    EXPECT_FALSE(a.has_value());

    // emplace new object in the move_only_any
    a.emplace<std::vector<int>>({ 0, 1, 2, 3 });

    // the move_only_any object must contain a string containing ten 'b' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a), (std::vector<int>{ 0, 1, 2, 3 }));
}

TEST(MoveOnlyAny, Reset) {
    // create move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // the constructed any should contain an object
    EXPECT_TRUE(a.has_value());

    // reset the move_only_any
    a.reset();

    // now, the move_only_any object should not contain an object anymore
    EXPECT_FALSE(a.has_value());
}

TEST(MoveOnlyAny, SwapMemberFunction) {
    // create two move_only_any objects
    plssvm::detail::move_only_any a1{ 42 };
    plssvm::detail::move_only_any a2{ 3.1415 };

    // swap both any objects
    a1.swap(a2);

    // check whether the content changed
    EXPECT_EQ(plssvm::detail::move_only_any_cast<double>(a1), 3.1415);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a2), 42);
}

TEST(MoveOnlyAny, HasValue) {
    // create move_only_any object that should contain an object
    const plssvm::detail::move_only_any a1{ 42 };
    EXPECT_TRUE(a1.has_value());

    // default constructed move_only_any should not contain an object
    const plssvm::detail::move_only_any a2{};
    EXPECT_FALSE(a2.has_value());
}

TEST(MoveOnlyAny, Type) {
    // default constructed move_only_any should return the typeid(void) on a call to .type()
    const plssvm::detail::move_only_any a1{};
    EXPECT_EQ(a1.type(), typeid(void));

    // normal constructed move_only_any should return the typeid of the contained type on a call to .type()
    const plssvm::detail::move_only_any a2{ 42 };
    EXPECT_EQ(a2.type(), typeid(int));
}

TEST(MoveOnlyAny, SwapFreeFunction) {
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

TEST(MoveOnlyAny, CastConstLvalueReference) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a), 42);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const int &>(a), 42);
}

TEST(MoveOnlyAny, CastConstLvalueReferenceWrongType) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<const float &>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, CastLvalueReference) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a), 42);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const int &>(a), 42);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int &>(a), 42);
}

TEST(MoveOnlyAny, CastLvalueReferenceWrongType) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<const float &>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float &>(a), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, CastRvalueReference) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ std::string{ "Hello, World!" } };
    // retrieve the contained value and check for correctness
    auto str = plssvm::detail::move_only_any_cast<std::string &&>(std::move(a));
    EXPECT_EQ(str, (std::string{ "Hello, World!" }));
}

TEST(MoveOnlyAny, CastRvalueReferenceWrongType) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::move_only_any_cast<float &&>(std::move(a)), plssvm::detail::bad_move_only_any_cast, "plssvm::detail::bad_move_only_any_cast");
}

TEST(MoveOnlyAny, CastConstPointer) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    const auto *ptr = plssvm::detail::move_only_any_cast<const int>(&a);
    EXPECT_EQ(*ptr, 42);
}

TEST(MoveOnlyAny, CastConstNullptrPointer) {
    // casting a nullptr should return a nullptr
    const plssvm::detail::move_only_any *a{ nullptr };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const int>(a), nullptr);
}

TEST(MoveOnlyAny, CastConstPointerWrongType) {
    // create const move_only_any object
    const plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const float>(&a), nullptr);
}

TEST(MoveOnlyAny, CastPointer) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // retrieve the contained value and check for correctness
    EXPECT_EQ(*plssvm::detail::move_only_any_cast<int>(&a), 42);
    EXPECT_EQ(*plssvm::detail::move_only_any_cast<const int>(&a), 42);
}

TEST(MoveOnlyAny, CastNullptrPointer) {
    // casting a nullptr should return a nullptr
    plssvm::detail::move_only_any *a{ nullptr };
    EXPECT_EQ(plssvm::detail::move_only_any_cast<int>(a), nullptr);
}

TEST(MoveOnlyAny, CastPointerWrongType) {
    // create const move_only_any object
    plssvm::detail::move_only_any a{ 42 };
    // try retrieving a value with the wrong type
    EXPECT_EQ(plssvm::detail::move_only_any_cast<float>(&a), nullptr);
    EXPECT_EQ(plssvm::detail::move_only_any_cast<const float>(&a), nullptr);
}

TEST(MoveOnlyAny, MakeMoveOnlyAny) {
    // construct a move_only_any object
    const auto a = plssvm::detail::make_move_only_any<std::string>(10, 'a');

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::string>(a), std::string{ "aaaaaaaaaa" });
}

TEST(MoveOnlyAny, MakeMoveOnlyAnyWithInitializerList) {
    // construct a move_only_any object
    const auto a = plssvm::detail::make_move_only_any<std::vector<int>>({ 0, 1, 2, 3 });

    // the move_only_any object must contain a string containing ten 'a' characters
    EXPECT_EQ(plssvm::detail::move_only_any_cast<std::vector<int>>(a), (std::vector<int>{ 0, 1, 2, 3 }));
}

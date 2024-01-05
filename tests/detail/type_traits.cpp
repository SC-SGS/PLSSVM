/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the type_traits header.
 */

#include "plssvm/detail/type_traits.hpp"

#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{HasSubstr, ContainsRegex}
#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <array>          // std::array
#include <deque>          // std::deque
#include <forward_list>   // std::forward_list
#include <list>           // std::list
#include <map>            // std::map, std::multimap
#include <set>            // std::set, std::multiset
#include <string>         // std::basic_string
#include <type_traits>    // std::is_same_v
#include <unordered_map>  // std::unordered_map, std::unordered_multimap
#include <unordered_set>  // std::unordered_set, std::unordered_multiset
#include <vector>         // std::vector

TEST(TypeTraits, always_false) {
    EXPECT_FALSE(plssvm::detail::always_false_v<void>);
    EXPECT_FALSE(plssvm::detail::always_false_v<int>);
    EXPECT_FALSE(plssvm::detail::always_false_v<double>);
}

TEST(TypeTraits, remove_cvref_t) {
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<volatile double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const volatile double>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<double &>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const double &>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<volatile double &>>) );
    EXPECT_TRUE((std::is_same_v<double, plssvm::detail::remove_cvref_t<const volatile double &>>) );
}

TEST(TypeTraits, is_string) {
    // sequence containers
    EXPECT_TRUE((plssvm::detail::is_string_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_string_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_string_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_string_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_string_v<int[2]>) );
}

TEST(TypeTraits, is_array) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_array_v<std::string>) );
    EXPECT_TRUE((plssvm::detail::is_array_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_array_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_array_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_array_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_array_v<int[2]>) );
}

TEST(TypeTraits, is_vector) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::array<int, 2>>) );
    EXPECT_TRUE((plssvm::detail::is_vector_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_vector_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_vector_v<int[2]>) );
}

TEST(TypeTraits, is_deque) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::vector<int>>) );
    EXPECT_TRUE((plssvm::detail::is_deque_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_deque_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_deque_v<int[2]>) );
}

TEST(TypeTraits, is_forward_list) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::deque<int>>) );
    EXPECT_TRUE((plssvm::detail::is_forward_list_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_forward_list_v<int[2]>) );
}

TEST(TypeTraits, is_list) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_list_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::forward_list<int>>) );
    EXPECT_TRUE((plssvm::detail::is_list_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_list_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_list_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_list_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_list_v<int[2]>) );
}

TEST(TypeTraits, is_set) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_set_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::list<int>>) );
    // associative containers
    EXPECT_TRUE((plssvm::detail::is_set_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_set_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_set_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_set_v<int[2]>) );
}

TEST(TypeTraits, is_map) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_map_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_map_v<std::set<int>>) );
    EXPECT_TRUE((plssvm::detail::is_map_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_map_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_map_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_map_v<int[2]>) );
}

TEST(TypeTraits, is_multiset) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::map<int, int>>) );
    EXPECT_TRUE((plssvm::detail::is_multiset_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multiset_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_multiset_v<int[2]>) );
}

TEST(TypeTraits, is_multimap) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::multiset<int>>) );
    EXPECT_TRUE((plssvm::detail::is_multimap_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_multimap_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_multimap_v<int[2]>) );
}

TEST(TypeTraits, is_unordered_set) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_TRUE((plssvm::detail::is_unordered_set_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_unordered_set_v<int[2]>) );
}

TEST(TypeTraits, is_unordered_map) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::unordered_set<int>>) );
    EXPECT_TRUE((plssvm::detail::is_unordered_map_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_unordered_map_v<int[2]>) );
}

TEST(TypeTraits, is_unordered_multiset) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::unordered_map<int, int>>) );
    EXPECT_TRUE((plssvm::detail::is_unordered_multiset_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_unordered_multiset_v<int[2]>) );
}

TEST(TypeTraits, is_unordered_multimap) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<std::unordered_multiset<int>>) );
    EXPECT_TRUE((plssvm::detail::is_unordered_multimap_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_unordered_multimap_v<int[2]>) );
}

TEST(TypeTraits, is_contiguous_container) {
    // sequence containers
    EXPECT_TRUE((plssvm::detail::is_contiguous_container_v<std::string>) );
    EXPECT_TRUE((plssvm::detail::is_contiguous_container_v<std::array<int, 2>>) );
    EXPECT_TRUE((plssvm::detail::is_contiguous_container_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_contiguous_container_v<int[2]>) );
}

TEST(TypeTraits, is_sequence_container) {
    // sequence containers
    EXPECT_TRUE((plssvm::detail::is_sequence_container_v<std::string>) );
    EXPECT_TRUE((plssvm::detail::is_sequence_container_v<std::array<int, 2>>) );
    EXPECT_TRUE((plssvm::detail::is_sequence_container_v<std::vector<int>>) );
    EXPECT_TRUE((plssvm::detail::is_sequence_container_v<std::deque<int>>) );
    EXPECT_TRUE((plssvm::detail::is_sequence_container_v<std::forward_list<int>>) );
    EXPECT_TRUE((plssvm::detail::is_sequence_container_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_sequence_container_v<int[2]>) );
}

TEST(TypeTraits, is_associative_container) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::list<int>>) );
    // associative containers
    EXPECT_TRUE((plssvm::detail::is_associative_container_v<std::set<int>>) );
    EXPECT_TRUE((plssvm::detail::is_associative_container_v<std::map<int, int>>) );
    EXPECT_TRUE((plssvm::detail::is_associative_container_v<std::multiset<int>>) );
    EXPECT_TRUE((plssvm::detail::is_associative_container_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::unordered_set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::unordered_map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::unordered_multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_associative_container_v<int[2]>) );
}

TEST(TypeTraits, is_unordered_associative_container) {
    // sequence containers
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::string>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::array<int, 2>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::vector<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::deque<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::forward_list<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::list<int>>) );
    // associative containers
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::set<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::map<int, int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::multiset<int>>) );
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_TRUE((plssvm::detail::is_unordered_associative_container_v<std::unordered_set<int>>) );
    EXPECT_TRUE((plssvm::detail::is_unordered_associative_container_v<std::unordered_map<int, int>>) );
    EXPECT_TRUE((plssvm::detail::is_unordered_associative_container_v<std::unordered_multiset<int>>) );
    EXPECT_TRUE((plssvm::detail::is_unordered_associative_container_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_unordered_associative_container_v<int[2]>) );
}

TEST(TypeTraits, is_container) {
    // sequence containers
    EXPECT_TRUE((plssvm::detail::is_container_v<std::string>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::array<int, 2>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::vector<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::deque<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::forward_list<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::list<int>>) );
    // associative containers
    EXPECT_TRUE((plssvm::detail::is_container_v<std::set<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::map<int, int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::multiset<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::multimap<int, int>>) );
    // unordered associative containers
    EXPECT_TRUE((plssvm::detail::is_container_v<std::unordered_set<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::unordered_map<int, int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::unordered_multiset<int>>) );
    EXPECT_TRUE((plssvm::detail::is_container_v<std::unordered_multimap<int, int>>) );
    // other
    EXPECT_FALSE((plssvm::detail::is_container_v<int[2]>) );
}

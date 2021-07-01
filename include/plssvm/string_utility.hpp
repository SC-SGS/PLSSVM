#pragma once

#include <fast_float/fast_float.h>
#include <fmt/core.h>

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>

namespace plssvm::util {

bool starts_with(const std::string_view str, const std::string_view start) noexcept {
  return str.substr(0, start.size()) == start;
}
bool starts_with(const std::string_view str, const char start) noexcept {
  return !str.empty() && std::char_traits<char>::eq(str.front(), start);
}
bool ends_with(const std::string_view str, const std::string_view end) noexcept {
  return str.size() >= end.size() && str.compare(str.size() - end.size(), std::string_view::npos, end) == 0;
}
bool ends_with(const std::string_view str, const char end) noexcept {
  return !str.empty() && std::char_traits<char>::eq(str.back(), end);
}

std::string_view trim_left(const std::string_view str) noexcept {
  std::size_t pos = std::min(str.find_first_not_of(' '), str.size());
  return str.substr(pos);
}

template <typename T, typename Exception = std::runtime_error>
T convert_to(std::string_view str) {
  // remove leading whitespaces
  str = trim_left(str);

  // convert string to float or integer
  if constexpr (std::is_floating_point_v<T>) {
    T val;
    auto res = fast_float::from_chars(str.data(), str.data() + str.size(), val);
    if (res.ec != std::errc{}) {
      throw Exception{fmt::format("Can't convert '{}' to a floating point value!", str)};
    }
    return val;
  } else if constexpr (std::is_integral_v<T>) {
    T val;
    auto res = std::from_chars(str.data(), str.data() + str.size(), val);
    if (res.ec != std::errc{}) {
      throw Exception{fmt::format("Can't convert '{}' to an integral value!", str)};
    }
    return val;
  } else {
    static_assert(!std::is_same_v<T, T>, "Can only convert arithmetic types!");
  }
}

}
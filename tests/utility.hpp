/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for testing.
 */

#ifndef PLSSVM_TESTS_UTILITY_HPP_
#define PLSSVM_TESTS_UTILITY_HPP_
#pragma once

#include "plssvm/parameter.hpp"  // plssvm::parameter

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // EXPECT_NE, ASSERT_FALSE

#ifdef __unix__
    #include <cstdlib>  // mkstemp
#else
    #include <fstream>  // std::ofstream
#endif

#include <algorithm>    // std::generate
#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::temp_directory_path, std::filesystem::exists, std::filesystem::remove
#include <iostream>     // std::cout
#include <limits>       // std::numeric_limits{max, lowest}
#include <random>       // std::random_device, std::mt19937, std::uniform_real_distribution
#include <sstream>      // std:stringstream, std::ostringstream, std::istringstream
#include <streambuf>    // std::streambuf
#include <string>       // std::string
#include <tuple>        // std::tuple, std::make_tuple, std::get
#include <type_traits>  // std::is_floating_point_v
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

namespace util {

/**
 * @brief Class used to redirect the standard output inside test cases.
 */
class redirect_output {
  public:
    /**
     * @brief Redirect the output and store the original standard output location.
     */
    redirect_output() {
        // capture std::cout
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());
    }
    /**
     * @brief Restore the original standard output location.
     */
    virtual ~redirect_output() {
        // end capturing std::cout
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};

/**
 * @brief A class encapsulating unique temporary file's name.
 * @details On UNIX systems use `mkstemp` to create a unique file in the temporary directory. On non-UNIX system create
 *          a file in the current directory using a random `unsigned long long` number.
 */
class temporary_file {
  public:
    /**
     * @brief Create a new unique temporary file.
     */
    temporary_file() {
#ifdef __unix__
        filename = std::filesystem::temp_directory_path().string();
        filename += "/tmpfile_XXXXXX";
        // create unique temporary file
        const int file_descriptor = mkstemp(filename.data());
        // immediately close file if possible
        if (file_descriptor >= 0) {
            close(file_descriptor);
        }
#else
        std::random_device device;
        std::mt19937 gen(device());
        std::uniform_int_distribution<unsigned long long> dist;
        filename = fmt::format("tmpfile_{}", dist(gen));
        while (std::filesystem::exists(std::filesystem::temp_directory_path() / filename)) {
            filename = fmt::format("tmpfile_{}", dist(gen));
        }
        // create file
        std::ofstream{ std::filesystem::temp_directory_path() / filename };
#endif
    }
    /**
     * @brief Remove the temporary file if it exists.
     */
    virtual ~temporary_file() {
        std::filesystem::remove(filename);
    }

    std::string filename;
};

/**
 * @brief Convert the parameter @p value to a std::string using a std::ostringstream.
 * @details Calls `ASSERT_FALSE` if the @p value couldn't be converted to its std::string representation.
 * @tparam T the type of the value that should be converted to a std::string
 * @param[in] value the value to convert
 * @return the std::string representation of @p value (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string convert_to_string(const T &value) {
    std::ostringstream output;
    output << value;
    // test if output was successful
    [&]() {
        // need immediate invoked lambda because of void return in ASSERT_FALSE
        ASSERT_FALSE(output.fail());
    }();
    return output.str();
}
/**
 * @brief Convert the std::string @p str to a value of type T using a std::istringstream.
 * @details Calls `ASSERT_FALSE` if the @p str couldn't be converted to a value of type @p T.
 * @tparam T the type of the value to which the std::string should be converted
 * @param[in] str the std::string to convert
 * @return the value represented by @p value (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T convert_from_string(const std::string &str) {
    std::istringstream input{ str };
    T value{};
    input >> value;
    // test if input was successful
    [&]() {
        // need immediate invoked lambda because of void return in ASSERT_FALSE
        ASSERT_FALSE(input.fail());
    }();
    return value;
}

/**
 * @brief Generate vector of @p size filled with random floating point values in the range [@p lower, @p upper].
 * @tparam T the type of the elements in the vector (must be a floating point type)
 * @param[in] size the size of the vector
 * @param[in] lower the lower bound of the random values in the vector
 * @param[in] upper the upper bound of the random values in the vector
 * @return the randomly generated vector (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> generate_random_vector(const std::size_t size, const T lower = T{ -1.0 }, const T upper = T{ 1.0 }) {
    static_assert(std::is_floating_point_v<T>, "Can only meaningfully use a uniform_real_distribution with a floating point type!");

    std::vector<T> vec(size);

    // fill vectors with random values
    static std::random_device device;
    static std::mt19937 gen(device());
    std::uniform_real_distribution<T> dist(lower, upper);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });

    return vec;
}

/**
 * @brief Scale the @p data set to the range [@p lower, @p upper].
 * @tparam T the type of the data that should be scaled (must be a floating point type)
 * @param[in] data the data to scale
 * @param[in] lower the lower bound to which the data should be scaled
 * @param[in] upper the upper bound to which the data should be scaled
 * @return a pair consisting of: [the data set scaled to [@p lower, @p upper], the scaling factors used to scale the data] (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::pair<std::vector<std::vector<T>>, std::vector<std::tuple<std::size_t, T, T>>> scale(const std::vector<std::vector<T>> &data, const T lower, const T upper) {
    static_assert(std::is_floating_point_v<T>, "Scaling a data set only makes sense for values with a floating point type!");

    std::vector<std::tuple<std::size_t, T, T>> factors(data.front().size(), std::make_tuple(0, std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()));
    // calculate the scaling factors
    for (std::size_t i = 0; i < factors.size(); ++i) {
        std::get<0>(factors[i]) = i;
        for (std::size_t j = 0; j < data.size(); ++j) {
            std::get<1>(factors[i]) = std::min(std::get<1>(factors[i]), data[j][i]);
            std::get<2>(factors[i]) = std::max(std::get<2>(factors[i]), data[j][i]);
        }
    }
    // scale the data set
    std::vector<std::vector<T>> ret = data;
    for (std::size_t i = 0; i < ret.size(); ++i) {
        for (std::size_t j = 0; j < ret.front().size(); ++j) {
            ret[i][j] = lower + (upper - lower) * (data[i][j] - std::get<1>(factors[j])) / (std::get<2>(factors[j]) - std::get<1>(factors[j]));
        }
    }
    return std::make_pair(std::move(ret), std::move(factors));
}

}  // namespace util

#endif  // PLSSVM_TESTS_UTILITY_HPP_
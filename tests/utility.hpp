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

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::replace_all
#include "plssvm/detail/type_traits.hpp"     // plssvm::detail::always_false_v
#include "plssvm/parameter.hpp"              // plssvm::parameter, plssvm::detail::parameter

#include "fmt/core.h"                        // fmt::format
#include "gtest/gtest.h"                     // FAIL

#ifdef __unix__
    #include <cstdlib>  // mkstemp
#endif

#include <algorithm>    // std::generate, std::min, std::max
#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::{temp_directory_path, exists, remove}
#include <fstream>      // std::ifstream, std::ofstream
#include <iostream>     // std::ostream, std::cout
#include <iterator>     // std::istreambuf_iterator
#include <limits>       // std::numeric_limits{max, lowest}
#include <random>       // std::random_device, std::mt19937, std::uniform_real_distribution
#include <sstream>      // std:stringstream, std::ostringstream, std::istringstream
#include <streambuf>    // std::streambuf
#include <string>       // std::string
#include <tuple>        // std::tuple, std::make_tuple, std::get, std::tuple_size
#include <type_traits>  // std::is_floating_point_v, std::is_same_v, std::is_signed_v, std::is_unsigned_v, std::decay_t
#include <utility>      // std::pair, std::make_pair, std::move, std::make_index_sequence, std::index_sequence
#include <vector>       // std::vector

namespace util {

/**
 * @brief Class used to redirect the output of the std::ostream @p out inside test cases.
 * @tparam out the output-stream to capture
 */
template <std::ostream *out = &std::cout>
class redirect_output {
  public:
    /**
     * @brief Redirect the output and store the original output location of @p out.
     */
    redirect_output() : sbuf_{ out->rdbuf() } {
        // capture the output from the out stream
        out->rdbuf(buffer_.rdbuf());
    }
    /**
     * @brief Copy-construction is unnecessary.
     */
    redirect_output(const redirect_output &) = delete;
    /**
     * @brief Move-construction is unnecessary.
     */
    redirect_output(redirect_output &&) = delete;
    /**
     * @brief Copy-assignment is unnecessary.
     */
    redirect_output &operator=(const redirect_output &) = delete;
    /**
     * @brief Move-assignment is unnecessary.
     */
    redirect_output &operator=(redirect_output &&) = delete;
    /**
     * @brief Restore the original output location of @p out.
     */
    virtual ~redirect_output() {
        // end capturing the output from the out stream
        out->rdbuf(sbuf_);
        sbuf_ = nullptr;
    }
    /**
     * @brief Return the captured content.
     * @return the captured content (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string get_capture() {
        // be sure that every captured output is available
        buffer_.flush();
        return buffer_.str();
    }

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};

/**
 * @brief A class encapsulating an unique temporary file's name.
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
     * @brief Copy-construction is unnecessary.
     */
    temporary_file(const temporary_file &) = delete;
    /**
     * @brief Move-construction is unnecessary.
     */
    temporary_file(temporary_file &&) = delete;
    /**
     * @brief Copy-assignment is unnecessary.
     */
    temporary_file &operator=(const temporary_file &) = delete;
    /**
     * @brief Move-assignment is unnecessary.
     */
    temporary_file &operator=(temporary_file &&) = delete;
    /**
     * @brief Remove the temporary file if it exists.
     */
    virtual ~temporary_file() {
        std::filesystem::remove(filename);
    }

    std::string filename{};
};

/**
 * @brief Convert the std::string @p str to a value of type T using a std::istringstream.
 * @tparam T the type of the value to which the std::string should be converted
 * @param[in] str the std::string to convert
 * @return the value represented by @p value (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T convert_from_string(const std::string &str) {
    std::istringstream input{ str };
    T value{};
    input >> value;
    return value;
}

/**
 * @brief Get two distinct labels based on the provided label type.
 * @details The distinct label values must be provided in increasing order (for a defined order in `std::map`).
 * @tparam T the label type
 * @return two distinct label (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::pair<T, T> get_distinct_label() {
    if constexpr (std::is_same_v<T, bool>) {
        return std::make_pair(false, true);
    } else if constexpr (sizeof(T) == sizeof(char)) {
        return std::make_pair('a', 'b');
    } else if constexpr (std::is_signed_v<T>) {
        return std::make_pair(-1, 1);
    } else if constexpr (std::is_unsigned_v<T>) {
        return std::make_pair(1, 2);
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::make_pair(-1.5, 1.5);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::make_pair("cat", "dog");
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "Unknown label type provided!");
    }
}

/**
 * @brief Replace the label placeholders in @p template_filename with the labels based on the template type @p T and
 *        write the instantiated template file to @p output_filename.
 * @tparam T the type of the labels to instantiate the file for
 * @param[in] template_filename the file used as template
 * @param[in] output_filename the file to save the instantiate template to
 */
template <typename T>
inline void instantiate_template_file(const std::string &template_filename, const std::string &output_filename) {
    // check whether the template_file exists
    if (!std::filesystem::exists(template_filename)) {
        FAIL() << fmt::format("The template file {} does not exist!", template_filename);
    }
    // get a label pair based on the current label type
    const auto [first_label, second_label] = util::get_distinct_label<T>();
    // read the data set template and replace the label placeholder with the correct labels
    std::ifstream input{ template_filename };
    std::string str((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    plssvm::detail::replace_all(str, "LABEL_1_PLACEHOLDER", fmt::format("{}", first_label));
    plssvm::detail::replace_all(str, "LABEL_2_PLACEHOLDER", fmt::format("{}", second_label));
    // write the data set with the correct labels to the temporary file
    std::ofstream out{ output_filename };
    out << str;
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

/**
 * @brief Construct an instance of type @p T using @p params and the values in the std::tuple @p tuple where all values in @p tuple are saved as a std::pair
 *        containing the named-parameter name and value.
 * @tparam T the type to construct
 * @tparam Tuple the tuple type used to construct @p T
 * @tparam Is an index sequence used to iterate through the @p tuple
 * @param params the SVM parameter used to construct @p T
 * @param tuple the tuple values used to construct @p T
 * @return an instance of type @p T (`[[nodiscard]]`)
 */
template <typename T, typename Tuple, size_t... Is>
[[nodiscard]] inline T construct_from_tuple(const plssvm::parameter &params, Tuple &&tuple, std::index_sequence<Is...>) {
    return T{ params, (std::get<Is>(tuple).first = std::get<Is>(tuple).second)... };
}
/**
 * @brief Construct an instance of type @p T using @p params and the values in the std::tuple @p tuple where all values in @p tuple are saved as a std::pair
 *        containing the named-parameter name and value.
 * @tparam T the type to construct
 * @tparam real_type the floating point type used for the SVM parameter
 * @tparam Tuple the tuple type used to construct @p T
 * @param params the SVM parameter used to construct @p T
 * @param tuple the tuple values used to construct @p T
 * @return an instance of type @p T (`[[nodiscard]]`)
 */
template <typename T, typename real_type, typename Tuple>
[[nodiscard]] inline T construct_from_tuple(const plssvm::detail::parameter<real_type> &params, Tuple &&tuple) {
    return construct_from_tuple<T>(static_cast<plssvm::parameter>(params),
                                   std::forward<Tuple>(tuple),
                                   std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}

/**
 * @brief Construct an instance of type @p T using the values in the std::tuple @p tuple where all values in @p tuple are saved as a std::pair
 *        containing the named-parameter name and value.
 * @tparam T the type to construct
 * @tparam Tuple the tuple type used to construct @p T
 * @tparam Is an index sequence used to iterate through the @p tuple
 * @param tuple the tuple values used to construct @p T
 * @return an instance of type @p T (`[[nodiscard]]`)
 */
template <typename T, typename Tuple, size_t... Is>
[[nodiscard]] inline T construct_from_tuple(Tuple &&tuple, std::index_sequence<Is...>) {
    return T{ (std::get<Is>(tuple).first = std::get<Is>(tuple).second)... };
}
/**
 * @brief Construct an instance of type @p T using the values in the std::tuple @p tuple where all values in @p tuple are saved as a std::pair
 *        containing the named-parameter name and value.
 * @tparam T the type to construct
 * @tparam Tuple the tuple type used to construct @p T
 * @param tuple the tuple values used to construct @p T
 * @return an instance of type @p T (`[[nodiscard]]`)
 */
template <typename T, typename Tuple>
[[nodiscard]] inline T construct_from_tuple(Tuple &&tuple) {
    return construct_from_tuple<T>(std::forward<Tuple>(tuple),
                                   std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
}

}  // namespace util

#endif  // PLSSVM_TESTS_UTILITY_HPP_
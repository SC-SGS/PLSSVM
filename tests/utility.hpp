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

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name_v
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::replace_all
#include "plssvm/detail/type_traits.hpp"           // plssvm::detail::always_false_v
#include "plssvm/parameter.hpp"                    // plssvm::parameter

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // FAIL

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
#include <random>       // std::random_device, std::mt19937, std::uniform_real_distribution, std::uniform_int_distribution
#include <sstream>      // std:stringstream, std::ostringstream, std::istringstream
#include <stdexcept>    // std::runtime_error
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
    redirect_output() :
        sbuf_{ out->rdbuf() } {
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
    /**
     * @brief Clear the current capture buffer.
     */
    void clear_capture() {
        buffer_.clear();
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
        // append tmp dir to filename
        filename = std::filesystem::temp_directory_path().string() + filename;
        // create file
        std::ofstream{ filename };
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
 * @brief Get three distinct labels (two in case of `bool` and four in case of `std::string`) based on the provided label type.
 * @details The provided labels are sorted according to `std::less` to reflect the same order as in a `std::map`.
 * @tparam T the label type
 * @return the distinct labels (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> get_distinct_label() {
    std::vector<T> ret;
    if constexpr (std::is_same_v<T, bool>) {
        ret = std::vector<T>{ false, true };
    } else if constexpr (sizeof(T) == sizeof(char)) {
        ret = std::vector<T>{ 'a', 'b', 'c' };
    } else if constexpr (std::is_signed_v<T>) {
        ret = std::vector<T>{ -1, 1, 2 };
    } else if constexpr (std::is_unsigned_v<T>) {
        ret = std::vector<T>{ 1, 2, 3 };
    } else if constexpr (std::is_floating_point_v<T>) {
        ret = std::vector<T>{ -1.5, 1.5, 2.5 };
    } else if constexpr (std::is_same_v<T, std::string>) {
        ret = std::vector<T>{ "cat", "dog", "mouse", "bird" };
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "Unknown label type provided!");
    }

    // sort labels according to std::map order
    std::sort(ret.begin(), ret.end(), std::less<T>{});

    // be sure that at most four and at least two labels are used!
    if (ret.size() > 4 || ret.size() < 2) {
        std::string label_type_name{};
        if constexpr (std::is_same_v<T, std::string>) {
            label_type_name = "std::string";
        } else {
            label_type_name = plssvm::detail::arithmetic_type_name<T>();
        }
        throw std::runtime_error{ fmt::format("The least two and at most four different labels are supported, but {} are specified for the label_type '{}'!", ret.size(), label_type_name) };
    }

    return ret;
}

/**
 * @brief Get the number of distinct labels based on the provided label type.
 * @details Same as `util::get_distinct_label<T>().size()`.
 * @tparam T the label type
 * @return the number distinct labels (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::size_t get_num_classes() {
    return get_distinct_label<T>().size();
}

/**
 * @brief Replace the label placeholders in input @p template_filename with the labels based on the template type @p T and
 *        write the instantiated template file to @p output_filename.
 * @details The template files contain four label placeholder. If less than four distinct labels are given, the last placeholders share the same label.
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

    // get the distinct labels based on the current label type
    const std::vector<T> labels = util::get_distinct_label<T>();

    // read the data set template and replace the label placeholder with the correct labels
    std::ifstream input{ template_filename };
    std::string str((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    plssvm::detail::replace_all(str, "LABEL_PLACEHOLDER", fmt::format("{}", fmt::join(labels, ",")));
    plssvm::detail::replace_all(str, "LABEL_1_PLACEHOLDER", fmt::format("{}", labels[std::min<std::size_t>(0, labels.size() - 1)]));
    plssvm::detail::replace_all(str, "LABEL_2_PLACEHOLDER", fmt::format("{}", labels[std::min<std::size_t>(1, labels.size() - 1)]));
    plssvm::detail::replace_all(str, "LABEL_3_PLACEHOLDER", fmt::format("{}", labels[std::min<std::size_t>(2, labels.size() - 1)]));
    plssvm::detail::replace_all(str, "LABEL_4_PLACEHOLDER", fmt::format("{}", labels[std::min<std::size_t>(3, labels.size() - 1)]));

    // write the data set with the correct labels to the temporary file
    std::ofstream out{ output_filename };
    out << str;
}
/**
 * @brief Get the label vector that is described in the input data template files.
 * @details Works according to the `instantiate_template_file` function.
 * @tparam T the type of the labels
 * @return the correct label vector with respect to the input data template files (`[[nodiscard]]``)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> get_correct_data_file_labels() {
    // get the distinct labels based on the current label type
    const std::vector<T> labels = util::get_distinct_label<T>();
    // for LABEL_PLACEHOLDER: [ 1, 1, 2, 3, 2, 4 ]
    // if only two labels, e.g., [ -1, 1 ] are given, the output will look as follows: [ -1, -1, 1, 1, 1, 1 ]
    // clang-format off
    return std::vector<T>{ labels[std::min<std::size_t>(0, labels.size() - 1)], labels[std::min<std::size_t>(0, labels.size() - 1)],
                           labels[std::min<std::size_t>(1, labels.size() - 1)], labels[std::min<std::size_t>(2, labels.size() - 1)],
                           labels[std::min<std::size_t>(1, labels.size() - 1)], labels[std::min<std::size_t>(3, labels.size() - 1)] };
    // clang-format on
}

/**
 * @brief Calculate the label distribution necessary for LIBSVM's model file `num_sv` field.
 * @details Example: distinct labels = [ -1, 1 ] and num_labels = 5 would return [ 3, 2 ]
 * @tparam T the type of the labels
 * @param num_labels the total number of labels used in this distribution
 * @return the label distribution (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<std::size_t> get_correct_model_file_num_sv_per_class(const std::size_t num_labels = 6) {
    // get the distinct labels based on the current label type
    const std::vector<T> labels = util::get_distinct_label<T>();

    std::vector<std::size_t> distribution(labels.size(), num_labels / labels.size());
    const std::size_t remaining = num_labels % labels.size();
    for (std::size_t i = 0; i < remaining; ++i) {
        ++distribution[i];
    }

    return distribution;
}

/**
 * @brief Get the label vector that is described in the model template files.
 * @details Works according to the `instantiate_template_file` function.
 * @tparam T the type of the labels
 * @return the correct label vector with respect to the model template files (`[[nodiscard]]``)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> get_correct_model_file_labels() {
    // get the distinct labels based on the current label type
    const std::vector<T> labels = util::get_distinct_label<T>();
    // get the label distribution
    const std::vector<std::size_t> num_sv = get_correct_model_file_num_sv_per_class<T>();

    std::vector<T> res;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        for (std::size_t j = 0; j < num_sv[i]; ++j) {
            res.emplace_back(labels[i]);
        }
    }

    return res;
}

/**
 * @brief Generate a vector of @p size filled with random floating point values in the range [@p lower, @p upper].
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
 * @brief Generate matrix of size @p rows times @p cols filled with random floating point values in the range `[-1.0, 1.0)`.
 * @tparam matrix_type the type of the elements in the vector (must be a floating point type)
 * @param[in] rows the number of rows in the matrix
 * @param[in] cols the number of columns in the matrix
 * @return the randomly generated matrix (`[[nodiscard]]`)
 */
template <typename matrix_type, typename real_type = typename matrix_type::value_type>
[[nodiscard]] inline matrix_type generate_random_matrix(const std::size_t rows, const std::size_t cols) {
    static_assert(std::is_floating_point_v<real_type>, "Only floating point types are allowed!");

    // create random number generator
    static std::random_device device;
    static std::mt19937 gen(device());
    std::uniform_real_distribution<real_type> dist(real_type{ -1.0 }, real_type{ 1.0 });

    matrix_type matrix { rows, cols };
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            matrix(i, j) = dist(gen);
        }
    }

    return matrix;
}
/**
 * @brief Generate matrix of size (@p rows + @p row_padding) times (@p cols + @p col_padding) filled with random floating point values in the range `[-1.0, 1.0)`.
 *        The padding entries are set to `0`.
 * @tparam matrix_type the type of the elements in the vector (must be a floating point type)
 * @param[in] rows the number of rows in the matrix
 * @param[in] cols the number of columns in the matrix
 * @param[in] row_padding the number of padding entries per row in the matrix
 * @param[in] col_padding the number of padding entries per column in the matrix
 * @return the randomly generated matrix (`[[nodiscard]]`)
 */
template <typename matrix_type, typename real_type = typename matrix_type::value_type>
[[nodiscard]] inline matrix_type generate_random_matrix(const std::size_t rows, const std::size_t cols, const std::size_t row_padding, const std::size_t col_padding) {
    return matrix_type{ generate_random_matrix<matrix_type>(rows, cols), row_padding, col_padding };
}

/**
 * @brief Generate a matrix of size @p rows times @p cols filled with filled with values "(row + col) / 10.0".
 * @tparam matrix_type the type of the elements in the matrix (must be a floating point type)
 * @param[in] rows the number of rows in the matrix
 * @param[in] cols the number of columns in the matrix
 * @return the generated matrix (`[[nodiscard]]`)
 */
template <typename matrix_type>
[[nodiscard]] inline matrix_type generate_specific_matrix(const std::size_t rows, const std::size_t cols) {
    using real_type = typename matrix_type::value_type;
    static_assert(std::is_floating_point_v<real_type>, "Only floating point types are allowed!");

    matrix_type matrix { rows, cols };
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 1; j <= cols; ++j) {
            matrix(i, j - 1) = i + j / real_type{ 10.0 };
        }
    }

    return matrix;
}
/**
 * @brief Generate a matrix of size (@p rows + @p row_padding) times (@p cols + @p col_padding) filled with filled with values "row.(col +1)".
 *        The padding entries are set to `0`.
 * @details Example for row = 1 and cols = 3 is: [ 1.1, 1.2, 1.3 ].
 * @tparam matrix_type the type of the elements in the matrix (must be a floating point type)
 * @param[in] rows the number of rows in the matrix
 * @param[in] cols the number of columns in the matrix
 * @param[in] row_padding the number of padding entries per row in the matrix
 * @param[in] col_padding the number of padding entries per column in the matrix
 * @return the generated matrix (`[[nodiscard]]`)
 */
template <typename matrix_type>
[[nodiscard]] inline matrix_type generate_specific_matrix(const std::size_t rows, const std::size_t cols, const std::size_t row_padding, const std::size_t col_padding) {
    return matrix_type{ generate_specific_matrix<matrix_type>(rows, cols), row_padding, col_padding };
}

/**
 * @brief Generate a "sparse" matrix of size @p rows times @p cols filled with filled with values "row.(col +1)".
 *        In each row, approximately 50% of the values are replaced with zeros.
 * @details Example for row = 1 and cols = 3 is: [ 1.1, 0.0, 1.3 ].
 * @tparam matrix_type the type of the elements in the vector (must be a floating point type)
 * @param[in] rows the number of rows in the matrix
 * @param[in] cols the number of columns in the matrix
 * @return the generated sparse matrix (`[[nodiscard]]`)
 */
template <typename matrix_type>
[[nodiscard]] inline matrix_type generate_specific_sparse_matrix(const std::size_t rows, const std::size_t cols) {
    using real_type = typename matrix_type::value_type;
    static_assert(std::is_floating_point_v<real_type>, "Only floating point types are allowed!");

    // random number generate for range [ 0.0, 1.0 ]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::size_t> dis(0, cols - 1);

    // generate sparse matrix
    matrix_type matrix{ rows, cols };
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 1; j <= cols; ++j) {
            matrix(i, j - 1) = i + j / real_type{ 10.0 };
        }
        // remove half of the created values randomly
        for (std::size_t j = 0; j < cols / 2; ++j) {
            matrix(i, dis(gen)) = real_type{ 0.0 };
        }
    }

    return matrix;
}
/**
 * @brief Generate a "sparse" matrix of size (@p rows + @p row_padding) times (@p cols + @p col_padding) filled with filled with values "row.(col +1)".
 *        In each row, approximately 50% of the values are replaced with zeros. The padding entries are set to `0`.
 * @details Example for row = 1 and cols = 3 is: [ 1.1, 0.0, 1.3 ].
 * @tparam matrix_type the type of the elements in the vector (must be a floating point type)
 * @param[in] rows the number of rows in the matrix
 * @param[in] cols the number of columns in the matrix
 * @param[in] row_padding the number of padding entries per row in the matrix
 * @param[in] col_padding the number of padding entries per column in the matrix
 * @return the generated sparse matrix (`[[nodiscard]]`)
 */
template <typename matrix_type>
[[nodiscard]] inline matrix_type generate_specific_sparse_matrix(const std::size_t rows, const std::size_t cols, const std::size_t row_padding, const std::size_t col_padding) {
    return matrix_type{ generate_specific_sparse_matrix<matrix_type>(rows, cols), row_padding, col_padding };
}

/**
 * @brief Convert the provided 2D vector to a 1D vector.
 * @tparam T the type in the vector
 * @param[in] vec_2D the 2D input vector
 * @return the 1D result vector (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> flatten(const std::vector<std::vector<T>> &vec_2D) {
    std::vector<T> ret{};
    for (const std::vector<T> &vec : vec_2D) {
        ret.insert(ret.end(), vec.begin(), vec.end());
    }
    return ret;
}


/**
 * @brief Scale the @p data set to the range [@p lower, @p upper].
 * @tparam T the type of the data that should be scaled (must be a floating point type)
 * @tparam layout the memory layout used for the plssvm::matrix @p data
 * @param[in] data the data to scale
 * @param[in] lower the lower bound to which the data should be scaled
 * @param[in] upper the upper bound to which the data should be scaled
 * @return a pair consisting of: [the data set scaled to [@p lower, @p upper], the scaling factors used to scale the data] (`[[nodiscard]]`)
 */
template <typename T, plssvm::layout_type layout>
[[nodiscard]] inline std::pair<plssvm::matrix<T, layout>, std::vector<std::tuple<std::size_t, T, T>>> scale(const plssvm::matrix<T, layout> &data, const T lower, const T upper) {
    static_assert(std::is_floating_point_v<T>, "Scaling a data set only makes sense for values with a floating point type!");

    std::vector<std::tuple<std::size_t, T, T>> factors(data.num_cols(), std::make_tuple(0, std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()));
    // calculate the scaling factors
    for (std::size_t i = 0; i < factors.size(); ++i) {
        std::get<0>(factors[i]) = i;
        for (std::size_t j = 0; j < data.num_rows(); ++j) {
            std::get<1>(factors[i]) = std::min(std::get<1>(factors[i]), data(j, i));
            std::get<2>(factors[i]) = std::max(std::get<2>(factors[i]), data(j, i));
        }
    }
    // scale the data set
    plssvm::matrix<T, layout> ret = data;
    for (std::size_t i = 0; i < ret.num_rows(); ++i) {
        for (std::size_t j = 0; j < ret.num_cols(); ++j) {
            ret(i, j) = lower + (upper - lower) * (data(i, j) - std::get<1>(factors[j])) / (std::get<2>(factors[j]) - std::get<1>(factors[j]));
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
 * @tparam Tuple the tuple type used to construct @p T
 * @param params the SVM parameter used to construct @p T
 * @param tuple the tuple values used to construct @p T
 * @return an instance of type @p T (`[[nodiscard]]`)
 */
template <typename T, typename Tuple>
[[nodiscard]] inline T construct_from_tuple(const plssvm::parameter &params, Tuple &&tuple) {
    return construct_from_tuple<T>(params,
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
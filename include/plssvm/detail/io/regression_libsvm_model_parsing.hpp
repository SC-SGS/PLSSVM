/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the LIBSVM SVR model file for the regression task.
 */

#ifndef PLSSVM_DETAIL_IO_REGRESSION_LIBSVM_MODEL_PARSING_HPP_
#define PLSSVM_DETAIL_IO_REGRESSION_LIBSVM_MODEL_PARSING_HPP_
#pragma once

#include "plssvm/constants.hpp"                     // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/detail/assert.hpp"                 // PLSSVM_ASSERT
#include "plssvm/detail/io/file_reader.hpp"         // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_parsing.hpp"      // plssvm::detail::io::parse_libsvm_num_features
#include "plssvm/detail/logging.hpp"                // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"            // plssvm::memory_size, custom literals
#include "plssvm/detail/string_conversion.hpp"      // plssvm::detail::{convert_to, split_as}
#include "plssvm/detail/string_utility.hpp"         // plssvm::detail::{trim, trim_left, to_lower_case}
#include "plssvm/gamma.hpp"                         // plssvm::get_gamma_string
#include "plssvm/kernel_function_types.hpp"         // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                        // plssvm::soa_matrix
#include "plssvm/mpi/communicator.hpp"              // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                     // plssvm::parameter
#include "plssvm/shape.hpp"                         // plssvm::shape
#include "plssvm/verbosity_levels.hpp"              // plssvm::verbosity_level

#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/format.h"   // fmt::format_to, fmt::format
#include "fmt/os.h"       // fmt::ostream, fmt::output_file
#include "fmt/ranges.h"   // fmt::join

#include <algorithm>    //std::find_first_of, std::min
#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <exception>    // std::exception_ptr, std::exception, std::current_exception, std::rethrow_exception
#include <sstream>      // std::istringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief Parse the modified LIBSVM SVR model file header.
 * @details An example modified LIBSVM model file header for the linear kernel could look like
 * @code
 * svm_type c_svr
 * kernel_type linear
 * nr_class 2
 * rho 0.37330625882191915
 * total_sv 8
 * SV
 * @endcode
 * @param[in] lines the LIBSVM SVR model file header to parse
 * @throws plssvm::invalid_file_format_exception if an invalid 'svm_type' has been provided, i.e., 'svm_type' is not 'c_svr'
 * @throws plssvm::invalid_file_format_exception if an invalid 'kernel_type has been provided
 * @throws plssvm::invalid_file_format_exception if the number of support vectors ('total_sv') is zero
 * @throws plssvm::invalid_file_format_exception if not exactly one rho value has been provided
 * @throws plssvm::invalid_file_format_exception if an invalid header entry has been read
 * @throws plssvm::invalid_file_format_exception if the 'svm_type' is missing
 * @throws plssvm::invalid_file_format_exception if the 'kernel_type' is missing
 * @throws plssvm::invalid_file_format_exception if SVM parameter are explicitly provided that are not used in the give kernel (e.g., 'gamma' is provided for the 'linear' kernel)
 * @throws plssvm::invalid_file_format_exception if the total number of support vectors ('total_sv') is missing
 * @throws plssvm::invalid_file_format_exception if the value for rho is missing
 * @throws plssvm::invalid_file_format_exception if no support vectors have been provided in the data section
 * @attention The PLSSVM model file is currently not compatible with LIBSVM due to other "svm_type" entries.
 * @return [the SVM parameter; the value of rho; the number of header lines] (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::tuple<plssvm::parameter, std::vector<real_type>, std::size_t> parse_libsvm_model_header_regression(const std::vector<std::string_view> &lines) {
    // data to read
    plssvm::parameter params{};
    std::vector<real_type> rho{};
    std::size_t num_support_vectors{};

    // helper variables
    bool svm_type_set{ false };
    bool kernel_type_set{ false };
    bool nr_class_set{ false };
    bool total_sv_set{ false };
    bool rho_set{ false };
    bool degree_provided{ false };
    bool gamma_provided{ false };
    bool coef0_provided{ false };
    std::size_t nr_class{};

    // parse libsvm model file header
    std::size_t header_line = 0;
    {
        for (; header_line < lines.size(); ++header_line) {
            // get the current line and convert it to lower case
            std::string line{ detail::trim(lines[header_line]) };
            detail::to_lower_case(line);

            // separate value from model header entry
            std::string_view value{ line };
            value.remove_prefix(std::min(value.find_first_of(' '), value.size()));
            value = detail::trim(value);

            if (detail::starts_with(line, "svm_type")) {
                // svm_type must be c_svc
                if (value != "c_svr") {
                    throw invalid_file_format_exception{ fmt::format("Can only use c_svr as svm_type, but '{}' was given!", value) };
                }
                // read the svm_type
                svm_type_set = true;
            } else if (detail::starts_with(line, "kernel_type")) {
                // parse kernel_type, must be linear, polynomial or rbf
                std::istringstream iss{ std::string{ value } };
                iss >> params.kernel_type;
                if (iss.fail()) {
                    throw invalid_file_format_exception{ fmt::format("Unrecognized kernel type '{}'!", value) };
                }
                // read the kernel_type
                kernel_type_set = true;
            } else if (detail::starts_with(line, "gamma")) {
                // parse gamma -> always use the real_type std::variant member here!
                params.gamma = detail::convert_to<real_type>(value);
                gamma_provided = true;
            } else if (detail::starts_with(line, "degree")) {
                // parse degree
                params.degree = detail::convert_to<decltype(params.degree)>(value);
                degree_provided = true;
            } else if (detail::starts_with(line, "coef0")) {
                // parse coef0
                params.coef0 = detail::convert_to<decltype(params.coef0)>(value);
                coef0_provided = true;
            } else if (detail::starts_with(line, "nr_class")) {
                // number of classes must be greater or equal than two
                nr_class = detail::convert_to<unsigned long long>(value);
                // read the number of classes (number of different labels)
                nr_class_set = true;
            } else if (detail::starts_with(line, "total_sv")) {
                // the total number of support vectors must be greater than 0
                num_support_vectors = detail::convert_to<std::size_t>(value);
                if (num_support_vectors == 0) {
                    throw invalid_file_format_exception{ "The number of support vectors must be greater than 0!" };
                }
                // read the number of support vectors
                total_sv_set = true;
            } else if (detail::starts_with(line, "rho")) {
                // parse rho, required
                rho = detail::split_as<real_type>(value, ' ');
                if (rho.empty()) {
                    throw invalid_file_format_exception{ "At least one rho value must be set, but none was given!" };
                }
                // read the rho value
                rho_set = true;
            } else if (line == "sv") {
                // start parsing support vectors, required
                break;
            } else {
                throw invalid_file_format_exception{ fmt::format("Unrecognized header entry '{}'! Maybe SV is missing?", lines[header_line]) };
            }
        }
    }

    // additional sanity checks
    if (!svm_type_set) {
        throw invalid_file_format_exception{ "Missing svm_type!" };
    }
    if (!kernel_type_set) {
        throw invalid_file_format_exception{ "Missing kernel_type!" };
    }
    // check provided values based on kernel_type
    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            if (degree_provided) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the degree parameter which is not used in the linear kernel!" };
            }
            if (gamma_provided) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the gamma parameter which is not used in the linear kernel!" };
            }
            if (coef0_provided) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the coef0 parameter which is not used in the linear kernel!" };
            }
            break;
        case plssvm::kernel_function_type::polynomial:
            break;
        case plssvm::kernel_function_type::rbf:
        case plssvm::kernel_function_type::laplacian:
        case plssvm::kernel_function_type::chi_squared:
            if (degree_provided) {
                throw invalid_file_format_exception{ fmt::format("Explicitly provided a value for the degree parameter which is not used in the {} kernel!", params.kernel_type) };
            }
            if (coef0_provided) {
                throw invalid_file_format_exception{ fmt::format("Explicitly provided a value for the coef0 parameter which is not used in the {} kernel!", params.kernel_type) };
            }
            break;
        case plssvm::kernel_function_type::sigmoid:
            if (degree_provided) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the degree parameter which is not used in the sigmoid kernel!" };
            }
            break;
    }
    if (!nr_class_set) {
        throw invalid_file_format_exception{ "Missing number of different classes nr_class!" };
    }
    if (!total_sv_set) {
        throw invalid_file_format_exception{ "Missing total number of support vectors total_sv!" };
    }
    if (!rho_set) {
        throw invalid_file_format_exception{ "Missing rho values!" };
    }
    // number of different label numbers must match the number of classes
    if (nr_class != 2) {
        throw invalid_file_format_exception{ fmt::format("The number of classes (nr_class) is {}, but must be 2!", nr_class) };
    }
    // check if no support vectors are given
    if (header_line + 1 >= lines.size()) {
        throw invalid_file_format_exception{ "Can't parse file: no support vectors are given or SV is missing!" };
    }
    // check for the minimum number of required rho values
    if (rho.size() != 1) {
        throw invalid_file_format_exception{ fmt::format("Provided {} rho values but only one is needed!", rho.size()) };
    }
    // check whether the number of SV is correct
    if (lines.size() - (header_line + 1) != num_support_vectors) {
        throw invalid_file_format_exception{ fmt::format("Found {} support vectors, but it should be {}!", lines.size() - (header_line + 1), num_support_vectors) };
    }

    return std::make_tuple(params, rho, header_line + 1);
}

/**
 * @brief Parse all data points and weights (alpha values) using the file @p reader, ignoring all empty lines and lines starting with an `#`.
 * @details An example data section can look like
 * @code
 * 5.3748085208e-01 1:5.6909150126e-01 2:1.7385902768e-01 3:-1.2544805985e-01 4:-2.9571509449e-01
 * -8.4228254963e-01 1:9.5925668223e-01 2:-7.6818755962e-01 3:6.3621573833e-01 4:1.6085979487e-01
 * -7.0935195569e-01 1:5.4545142940e-02 2:2.8991780132e-01 3:7.2598021499e-01 4:-2.3469246049e-01
 * 3.8588219373e-01 1:3.6828886105e-01 2:8.0120546896e-01 3:6.1204205296e-01 4:-1.2074044818e-02
 * 5.5593569052e-01 1:-1.0777233473e-02 2:-2.0706684519e-01 3:6.6972496646e-01 4:-1.7887705784e-01
 * -2.2729386622e-01 1:2.9114472700e-01 2:3.1334616548e-02 3:5.5877309773e-01 4:2.4442300699e-01
 * 1.2048548972e+00 1:5.6054664148e-02 2:6.6037150647e-01 3:2.6221749918e-01 4:3.7216083001e-02
 * -6.9331414578e-01 1:2.3912649397e-02 2:3.4604367826e-01 3:-2.5696199624e-01 4:6.0591633490e-01
 * @endcode
 * @param[in] reader the file_reader used to read the LIBSVM data
 * @param[in] skipped_lines the number of lines that should be skipped at the beginning
 * @note The features must be provided with one-based indices!
 * @throws plssvm::invalid_file_format_exception if no features could be found (may indicate an empty file)
 * @throws plssvm::invalid_file_format_exception if not exactly one weight per class has been provided
 * @throws plssvm::invalid_file_format_exception if a weight couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if a feature index couldn't be converted to `unsigned long`
 * @throws plssvm::invalid_file_format_exception if a feature value couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if the provided LIBSVM file uses zero-based indexing (LIBSVM mandates one-based indices)
 * @throws plssvm::invalid_file_format_exception if the feature (indices) are not given in a strictly increasing order
 * @attention The PLSSVM model file is currently not compatible with LIBSVM due to other "svm_type" entries.
 * @return [the data points; the weights] (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::tuple<soa_matrix<real_type>, std::vector<aos_matrix<real_type>>> parse_libsvm_model_data_regression(const file_reader &reader, const std::size_t skipped_lines) {
    PLSSVM_ASSERT(reader.is_open(), "The file_reader is currently not associated with a file!");
    PLSSVM_ASSERT(skipped_lines <= reader.num_lines(), "Tried to skipp {} lines, but only {} are present!", skipped_lines, reader.num_lines());

    // parse sizes
    const std::size_t num_data_points = reader.num_lines() - skipped_lines;
    const std::size_t num_features = parse_libsvm_num_features(reader.lines(), skipped_lines);

    // no features were parsed -> invalid file
    if (num_features == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file: no data points are given!") };
    }

    // create vector containing the data and label
    soa_matrix<real_type> data{ shape{ num_data_points, num_features }, shape{ PADDING_SIZE, PADDING_SIZE } };
    aos_matrix<real_type> alpha{ shape{ 1, num_data_points }, shape{ PADDING_SIZE, PADDING_SIZE } };

    std::exception_ptr parallel_exception;

#pragma omp parallel default(none) shared(reader, skipped_lines, data, alpha, parallel_exception) firstprivate(num_data_points)
    {
#pragma omp for
        for (std::size_t i = 0; i < num_data_points; ++i) {
            try {
                const std::string_view line = reader.line(skipped_lines + i);
                unsigned long last_index = 0;

                // parse the alpha (weight) value -> must be exactly one for regression!
                std::string_view::size_type pos = 0;
                {
                    const std::string_view::size_type first_colon = line.find_first_of(":\n");
                    bool alpha_already_found = false;
                    while (true) {
                        const std::string_view::size_type next_pos = line.find_first_of(" \n", pos);
                        if (first_colon >= next_pos) {
                            if (alpha_already_found) {
                                throw invalid_file_format_exception{ "Can't parse file: needed exactly one alpha value, but more were provided!" };
                            }

                            // get alpha value
                            alpha(0, i) = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos));
                            pos = next_pos + 1;
                            alpha_already_found = true;
                        } else {
                            if (!alpha_already_found) {
                                throw invalid_file_format_exception{ "Can't parse file: needed exactly one alpha value, but none were provided!" };
                            }
                            break;
                        }
                    }
                }

                // get data
                while (true) {
                    std::string_view::size_type next_pos = line.find_first_of(':', pos);

                    // no further data points
                    if (next_pos == std::string_view::npos) {
                        break;
                    }

                    // get index
                    auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));

                    // LIBSVM assumes a 1-based indexing -> if the parsed index is 0 this condition is violated
                    if (index == 0) {
                        throw invalid_file_format_exception{ "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!" };
                    }
                    // the indices must be strictly increasing!
                    if (last_index >= index) {
                        throw invalid_file_format_exception{ fmt::format("The features indices must be strictly increasing, but {} is smaller or equal than {}!", index, last_index) };
                    }
                    last_index = index;

                    // since arrays start at 0, reduce 1 based index by one
                    --index;
                    pos = next_pos + 1;

                    // get value
                    next_pos = line.find_first_of(' ', pos);
                    data(i, index) = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                    pos = next_pos;
                }
            } catch (const std::exception &) {
                // catch first exception and store it
#pragma omp critical
                {
                    if (!parallel_exception) {
                        parallel_exception = std::current_exception();
                    }
                }
            }
        }
    }

    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }

    return std::make_tuple(std::move(data), std::vector<aos_matrix<real_type>>{ std::move(alpha) });
}

/**
 * @brief Write the modified LIBSVM SVC model file header to @p out.
 * @details An example modified LIBSVM SVC model file header for the linear kernel could look like
 * @code
 * svm_type c_svr
 * kernel_type linear
 * nr_class 2
 * rho 0.37330625882191915
 * total_sv 8
 * SV
 * @endcode
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in,out] out the output-stream to write the header information to
 * @param[in] comm the used MPI communicator
 * @param[in] params the SVM parameters
 * @param[in] rho the rho value
 * @param[in] data the data used to create the model
 * @attention The PLSSVM model file is currently not compatible with LIBSVM due to other "svm_type" entries.
 */
template <typename label_type>
inline void write_libsvm_model_header_regression(fmt::ostream &out, const mpi::communicator &comm, const plssvm::parameter &params, const std::vector<real_type> &rho, const regression_data_set<label_type> &data) {
    PLSSVM_ASSERT(rho.size() == 1, "Exactly one rho value must be provided!");

    // save model file header
    std::string out_string = fmt::format("svm_type c_svr\nkernel_type {}\n", params.kernel_type);
    // save the SVM parameter information based on the used kernel_type
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial:
            out_string += fmt::format("degree {}\ngamma {}\ncoef0 {}\n", params.degree, get_gamma_string(params.gamma), params.coef0);
            break;
        case kernel_function_type::rbf:
        case kernel_function_type::laplacian:
        case kernel_function_type::chi_squared:
            out_string += fmt::format("gamma {}\n", get_gamma_string(params.gamma));
            break;
        case kernel_function_type::sigmoid:
            out_string += fmt::format("\ngamma {}\ncoef0 {}\n", get_gamma_string(params.gamma), params.coef0);
            break;
    }

    out_string += fmt::format("nr_class 2\ntotal_sv {}\nrho {:.10e}\nSV\n",
                              data.num_data_points(),
                              fmt::join(rho, " "));

    // print model header
    detail::log(verbosity_level::full | verbosity_level::libsvm,
                comm,
                "\n{}\n",
                out_string);
    // write model header to file
    out.print("{}", out_string);
}

/**
 * @brief Write the modified LIBSVM SVC model to the file @p filename.
 * @details An example modified LIBSVM model file for the linear kernel could look like
 * @code
 * svm_type c_svr
 * kernel_type linear
 * nr_class 2
 * rho 0.37330625882191915
 * total_sv 8
 * SV
 * 5.3748085208e-01 1:5.6909150126e-01 2:1.7385902768e-01 3:-1.2544805985e-01 4:-2.9571509449e-01
 * -8.4228254963e-01 1:9.5925668223e-01 2:-7.6818755962e-01 3:6.3621573833e-01 4:1.6085979487e-01
 * -7.0935195569e-01 1:5.4545142940e-02 2:2.8991780132e-01 3:7.2598021499e-01 4:-2.3469246049e-01
 * 3.8588219373e-01 1:3.6828886105e-01 2:8.0120546896e-01 3:6.1204205296e-01 4:-1.2074044818e-02
 * 5.5593569052e-01 1:-1.0777233473e-02 2:-2.0706684519e-01 3:6.6972496646e-01 4:-1.7887705784e-01
 * -2.2729386622e-01 1:2.9114472700e-01 2:3.1334616548e-02 3:5.5877309773e-01 4:2.4442300699e-01
 * 1.2048548972e+00 1:5.6054664148e-02 2:6.6037150647e-01 3:2.6221749918e-01 4:3.7216083001e-02
 * -6.9331414578e-01 1:2.3912649397e-02 2:3.4604367826e-01 3:-2.5696199624e-01 4:6.0591633490e-01
 * @endcode
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in] filename the file to write the LIBSVM model to
 * @param[in] comm the used MPI communicator
 * @param[in] params the SVM parameters
 * @param[in] rho the rho value resulting from the hyperplane learning
 * @param[in] alpha the weights learned by the SVM
 * @param[in] data the data used to create the model
 * @attention The PLSSVM model file is only compatible with LIBSVM for the one vs. one classification type.
 */
template <typename label_type>
inline void write_libsvm_model_data_regression(const std::string &filename, const mpi::communicator &comm, const plssvm::parameter &params, const std::vector<real_type> &rho, const std::vector<aos_matrix<real_type>> &alpha, const regression_data_set<label_type> &data) {
    PLSSVM_ASSERT(!filename.empty(), "The provided model filename must not be empty!");
    PLSSVM_ASSERT(rho.size() == 1,
                  "The number of rho values is {} but must be exactly 1!",
                  rho.size());
#if defined(PLSSVM_ENABLE_ASSERTS)
    // weights
    PLSSVM_ASSERT(alpha.size() == 1, "The alpha vector may only contain one matrix as entry, but has {}!", alpha.size());
    PLSSVM_ASSERT(alpha.front().num_rows() == 1, "The number of rows in the matrix must be 1, but is {}!", alpha.front().num_rows());
    PLSSVM_ASSERT(alpha.front().num_cols() == data.num_data_points(), "The number of weights ({}) must be equal to the number of support vectors ({})!", alpha.front().num_cols(), data.num_data_points());
#endif
    using namespace literals;

    const soa_matrix<real_type> &support_vectors = data.data();
    const std::size_t num_features = data.num_features();

    // create file
    fmt::ostream out = fmt::output_file(filename);

    // write header information
    write_libsvm_model_header_regression(out, comm, params, rho, data);

    // the maximum size of one formatted LIBSVM entry, e.g., 1234:1.365363e+10
    // biggest number representable as std::size_t: 18446744073709551615 -> 20 chars
    // scientific notation: 3 chars (number in front of decimal separator including a sign + decimal separator) + 10 chars (part after the decimal separator, specified during formatting) +
    //                      5 chars exponent (e + sign + maximum potential exponent (308 -> 3 digits)
    // separators: 2 chars (: between index and feature + whitespace after feature value)
    // -> 40 chars in total
    // -> increased to 48 chars to be on the safe side
    constexpr static std::size_t CHARS_PER_BLOCK = 48;
    // results in 48 B * 128 B = 6 KiB stack buffer per thread
    constexpr static std::size_t BLOCK_SIZE = 128;
    // use 1 MiB as buffer per thread
    constexpr detail::memory_size STRING_BUFFER_SIZE = 1_MiB;

    // format one output-line
    auto format_libsvm_line = [](std::string &output, const real_type a, const soa_matrix<real_type> &d, const std::size_t point) {
        constexpr static std::size_t STACK_BUFFER_SIZE = BLOCK_SIZE * CHARS_PER_BLOCK;
        static std::array<char, STACK_BUFFER_SIZE> buffer{};
#pragma omp threadprivate(buffer)

        output.append(fmt::format("{:.10e} ", a));
        for (typename std::vector<real_type>::size_type j = 0; j < d.num_cols(); j += BLOCK_SIZE) {
            char *ptr = buffer.data();
            for (std::size_t i = 0; i < std::min<std::size_t>(BLOCK_SIZE, d.num_cols() - j); ++i) {
                if (d(point, j + i) != real_type{ 0.0 }) {
                    // add 1 to the index since LIBSVM assumes 1-based feature indexing
                    ptr = fmt::format_to(ptr, FMT_COMPILE("{}:{:.10e} "), j + i + 1, d(point, j + i));
                }
            }
            output.append(buffer.data(), static_cast<std::string::size_type>(ptr - buffer.data()));
        }
        output.push_back('\n');
    };

#pragma omp parallel default(none) shared(alpha, format_libsvm_line, support_vectors, out) firstprivate(STRING_BUFFER_SIZE, num_features)
    {
        // preallocate string buffer, only ONE allocation
        std::string out_string;
        out_string.reserve(STRING_BUFFER_SIZE.num_bytes() + ((num_features + std::size_t{ 1 }) * CHARS_PER_BLOCK));  // oversubscribe buffer that at least one additional line fits into it

#pragma omp for
        for (std::size_t i = 0; i < support_vectors.num_rows(); ++i) {
            // format the current LIBSVM line
            format_libsvm_line(out_string, alpha.front()(0, i), support_vectors, i);

            // if the buffer is full, write it to the file
            if (out_string.size() > STRING_BUFFER_SIZE.num_bytes()) {
#pragma omp critical
                {
                    out.print("{}", out_string);
                }
                // clear buffer
                out_string.clear();
            }
        }

#pragma omp critical
        {
            if (!out_string.empty()) {
                out.print("{}", out_string);
            }
        }
    }
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_REGRESSION_LIBSVM_MODEL_PARSING_HPP_

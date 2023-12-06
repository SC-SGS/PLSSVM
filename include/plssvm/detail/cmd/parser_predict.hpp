/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a class encapsulating all necessary parameters for predicting using the C-SVM possibly provided through command line arguments.
 */

#ifndef PLSSVM_DETAIL_CMD_PARSER_PREDICT_HPP_
#define PLSSVM_DETAIL_CMD_PARSER_PREDICT_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                      // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl::implementation_type
#include "plssvm/target_platforms.hpp"                   // plssvm::target_platform

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream
#include <string>  // std::string

namespace plssvm::detail::cmd {

/**
 * @brief Struct for encapsulating all necessary parameters for prediction; normally provided through command line arguments.
 */
struct parser_predict {
    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the predict parameters accordingly.
     * @details If no output filename is given, uses the input filename and appends a ".predict". The output file is than saved in the current working directory.
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parser_predict(int argc, char **argv);

    /// The used backend: automatic (depending on the specified target_platforms), OpenMP, CUDA, HIP, OpenCL, or SYCL.
    backend_type backend{ backend_type::automatic };
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD, or Intel.
    target_platform target{ target_platform::automatic };

    /// The SYCL implementation to use with `--backend sycl`.
    sycl::implementation_type sycl_implementation_type{ sycl::implementation_type::automatic };

    /// `true` if `std::string` should be used as label type instead of the default type `ìnt`.
    bool strings_as_labels{ false };

    /// The name of the data file to predict.
    std::string input_filename{};
    /// The name of the model file containing the support vectors and weights used for prediction.
    std::string model_filename{};
    /// The name of the file to write the predicted labels to.
    std::string predict_filename{};

    /// If performance tracking has been enabled, provides the name of the file where the performance tracking results are saved to. If the filename is empty, the results are dumped to stdout instead.
    std::string performance_tracking_filename{};
};

/**
 * @brief Output all predict parameters encapsulated by @p params to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the predict parameters to
 * @param[in] params the predict parameters
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const parser_predict &params);

}  // namespace plssvm::detail::cmd

template <>
struct fmt::formatter<plssvm::detail::cmd::parser_predict> : fmt::ostream_formatter {};

#endif  // PLSSVM_DETAIL_CMD_PARSER_PREDICT_HPP_
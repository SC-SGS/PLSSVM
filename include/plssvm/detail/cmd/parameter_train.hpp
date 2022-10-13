/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a class encapsulating all necessary parameters for training the C-SVM possibly provided through command line arguments.
 */

#ifndef PLSSVM_DETAIL_CMD_PARAMETER_TRAIN_HPP_
#define PLSSVM_DETAIL_CMD_PARAMETER_TRAIN_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/default_value.hpp"                         // plssvm::default_value
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include <cstddef>  // std::size_t
#include <iosfwd>   // forward declare std::ostream
#include <string>   // std::string

namespace plssvm::detail::cmd {

namespace sycl {
using namespace ::plssvm::sycl_generic;
}

/**
 * @brief Class for encapsulating all necessary parameters for training possibly provided through command line arguments.
 */
class parameter_train {
  public:
    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the training parameters accordingly. Parse the given data file.
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parameter_train(int argc, char **argv);

    /// Other base C-SVM parameters
    plssvm::parameter csvm_params{};

    /// The error tolerance parameter for the CG algorithm.
    default_value<double> epsilon{ default_init<double>{ 0.001 } };
    /// The maximum number of iterations in the CG algorithm.
    default_value<std::size_t> max_iter{ default_init<std::size_t>{ 0 } };

    /// The used backend: automatic (depending on the specified target_platforms), OpenMP, CUDA, HIP, OpenCL, or SYCL.
    backend_type backend{ backend_type::automatic };
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD or Intel.
    target_platform target{ target_platform::automatic };

    /// The kernel invocation type when using SYCL as backend.
    sycl::kernel_invocation_type sycl_kernel_invocation_type{ sycl::kernel_invocation_type::automatic };
    /// The SYCL implementation to use with --backend=sycl.
    sycl::implementation_type sycl_implementation_type{ sycl::implementation_type::automatic };

    /// `true` if `std::string` should be used as label type instead of the default type `ìnt`.
    bool strings_as_labels{ false };
    /// `true` if `float` should be used as real type instead of the default type `double`.
    bool float_as_real_type{ false };

    /// The name of the data/test file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned support vectors to/to parse the saved model from.
    std::string model_filename{};
};

/**
 * @brief Output all train parameters encapsulated by @p params to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameters
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const parameter_train &params);

}  // namespace plssvm::detail::cmd

#endif  // PLSSVM_DETAIL_CMD_PARAMETER_TRAIN_HPP_
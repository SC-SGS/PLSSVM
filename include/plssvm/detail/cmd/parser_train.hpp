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

#ifndef PLSSVM_DETAIL_CMD_PARSER_TRAIN_HPP_
#define PLSSVM_DETAIL_CMD_PARSER_TRAIN_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                        // plssvm::backend_type
#include "plssvm/backends/Kokkos/execution_spaces.hpp"     // plssvm::kokkos::execution_space
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/implementation_types.hpp"   // plssvm::sycl::implementation_type
#include "plssvm/classification_types.hpp"                 // plssvm::classification_type
#include "plssvm/constants.hpp"                            // plssvm::real_type, plssvm::DEFAULT_EPSILON
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                            // plssvm::parameter
#include "plssvm/solver_types.hpp"                         // plssvm::solving_type
#include "plssvm/svm_types.hpp"                            // plssvm::svm_type
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // mt::ostream_formatter

#include <cstddef>  // std::size_t
#include <iosfwd>   // forward declare std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::cmd {

/**
 * @brief Struct for encapsulating all necessary parameters for training; normally provided through command line arguments.
 */
struct parser_train {
    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the training parameters accordingly.
     * @details If no model filename is given, uses the input filename and appends a ".model". The model file is than saved in the current working directory.
     * @param[in] comm the MPI communicator wrapper
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parser_train(const mpi::communicator &comm, int argc, char **argv);

    /// Other base C-SVM parameters
    plssvm::parameter csvm_params{};

    /// The error tolerance parameter for the CG algorithm.
    real_type epsilon = static_cast<real_type>(1e-10);
    /// The maximum number of iterations in the CG algorithm.
    std::size_t max_iter{ 0 };
    /// The multi-class classification strategy used.
    classification_type classification{ classification_type::oaa };

    /// The used SVM: C-SVC or C-SVR.
    svm_type svm{ svm_type::csvc };
    /// The used backend: automatic (depending on the specified target_platforms), OpenMP, HPX, stdpar, CUDA, HIP, OpenCL, SYCL, or Kokkos.
    backend_type backend{ backend_type::automatic };
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD, or Intel.
    target_platform target{ target_platform::automatic };
    /// The used solver type for the LS-SVM kernel matrix: automatic (depending on the available (V)RAM), cg_explicit, or cg_implicit.
    solver_type solver{ solver_type::automatic };

    /// The data parallel kernel when using SYCL as backend.
    sycl::data_parallel_kernel sycl_data_parallel_kernel{ sycl::data_parallel_kernel::automatic };
    /// The SYCL implementation to use with --backend=sycl.
    sycl::implementation_type sycl_implementation_type{ sycl::implementation_type::automatic };

    /// The Kokkos execution space to use with --backend=kokkos.
    kokkos::execution_space kokkos_execution_space{ kokkos::execution_space::automatic };

    /// `true` if `std::string` should be used as label type for the classification task instead of the default type `ìnt`.
    /// For the regression task, this parameter is ignored and `real_type` is always used.
    bool strings_as_labels{ false };

    /// Load balancing weights for MPI used if different hardware per MPI process is used. The number must match the number of spawned MPI processes.
    /// Providing [1, 1] means every process gets the same amount of work, providing [1, 3] means that the second process has three times the work to do compared to process zero.
    std::vector<std::size_t> mpi_load_balancing_weights{};

    /// The name of the data/test file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned support vectors to/to parse the saved model from.
    std::string model_filename{};

    /// If performance tracking has been enabled, provides the name of the file where the performance tracking results are saved to. If the filename is empty, the results are dumped using std::clog instead.
    std::string performance_tracking_filename{};
};

/**
 * @brief Output all train parameters encapsulated by @p params to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the train parameters to
 * @param[in] params the train parameters
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const parser_train &params);

}  // namespace plssvm::detail::cmd

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::detail::cmd::parser_train> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_DETAIL_CMD_PARSER_TRAIN_HPP_

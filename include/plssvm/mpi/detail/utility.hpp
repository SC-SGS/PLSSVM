/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines some utility functions for our optional MPI usage.
 */

#ifndef PLSSVM_MPI_DETAIL_UTILITY_HPP_
#define PLSSVM_MPI_DETAIL_UTILITY_HPP_
#pragma once

#include <string>  // std::string

/**
 * @def PLSSVM_MPI_ERROR_CHECK
 * @brief Check the MPI error @p err. If @p err signals an error, throw a plssvm::mpi_exception.
 * @throws plssvm::mpi_exception if the error code signals a failure
 */
#if defined(PLSSVM_HAS_MPI_ENABLED)
    #define PLSSVM_MPI_ERROR_CHECK(err) plssvm::mpi::detail::mpi_error_check(err)
#else
    #define PLSSVM_MPI_ERROR_CHECK(...)
#endif

namespace plssvm::mpi::detail {

/**
 * @brief Checks whether @p err is equal to `MPI_SUCCESS`. If this is not the case, throws an exception.
 * @param[in] err the error code to check
 */
void mpi_error_check(int err);

/**
 * @brief Get the current processor name.
 * @return the processor name (`[[nodiscard]]`)
 */
[[nodiscard]] std::string node_name();

}  // namespace plssvm::mpi::detail

#endif  // PLSSVM_MPI_DETAIL_UTILITY_HPP_

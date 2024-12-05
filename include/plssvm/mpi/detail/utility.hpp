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

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception

#include "fmt/format.h"  // fmt::format

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_SUCCESS, MPI_MAX_ERROR_STRING, MPI_Error_string
#endif

#include <string>  // std::string

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #define PLSSVM_MPI_ERROR_CHECK(err)                                                                                              \
        if ((err) != MPI_SUCCESS) {                                                                                                  \
            std::string err_str(MPI_MAX_ERROR_STRING, '\0');                                                                         \
            int err_str_len{};                                                                                                       \
            const int res = MPI_Error_string(err, err_str.data(), &err_str_len);                                                     \
            if (res == MPI_SUCCESS) {                                                                                                \
                throw plssvm::mpi_exception{ fmt::format("MPI error {}: {}", err, err_str.substr(0, err_str.find_first_of('\0'))) }; \
            } else {                                                                                                                 \
                throw plssvm::mpi_exception{ fmt::format("MPI error {}", err) };                                                     \
            }                                                                                                                        \
        }
#else
    #define PLSSVM_MPI_ERROR_CHECK(...)
#endif

#endif  // PLSSVM_MPI_DETAIL_UTILITY_HPP_

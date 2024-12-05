/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines some wrapper functions necessary for our MPI support.
 * @details These wrapper functions are only conditionally compiled such that MPI is still an **optional** dependency in PLSSVM.
 */

#ifndef PLSSVM_DETAIL_MPI_WRAPPER_HPP_
#define PLSSVM_DETAIL_MPI_WRAPPER_HPP_
#pragma once

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_Comm, MPI_COMM_WORLD
#endif

#include <cstddef>  // std::size_t

namespace plssvm::detail::mpi {

class communicator {
  public:
    communicator();

#if defined(PLSSVM_HAS_MPI_ENABLED)
    communicator(MPI_Comm comm);
#endif

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t rank() const;
    [[nodiscard]] bool is_main_rank() const;

  private:
#if defined(PLSSVM_HAS_MPI_ENABLED)
    MPI_Comm comm_{ MPI_COMM_WORLD };
#endif
};

void init();
void init(int &argc, char **argv);
void finalize();

[[nodiscard]] bool is_initialized();
[[nodiscard]] bool is_finalized();

}  // namespace plssvm::detail::mpi

#endif  // PLSSVM_DETAIL_MPI_WRAPPER_HPP_

/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Defines a wrapper class around MPI's communicators.
* @details This wrapper class is only conditionally compiled such that MPI is still an **optional** dependency in PLSSVM.
*/

#ifndef PLSSVM_MPI_COMMUNICATOR_HPP_
#define PLSSVM_MPI_COMMUNICATOR_HPP_
#pragma once

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"
#endif

namespace plssvm::mpi {

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

}

#endif  // PLSSVM_MPI_COMMUNICATOR_HPP_

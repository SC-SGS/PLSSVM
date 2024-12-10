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
    #include "mpi.h"  // MPI_Comm, MPI_COMM_WORLD
#endif

#include <cstddef>  // std::size_t

namespace plssvm::mpi {

/**
 * @brief A small wrapper around MPI functions used in PLSSVM.
 * @details If PLSSVM was built without MPI support, this wrapper defines the respective functions to be essential no-ops.
 */
class communicator {
  public:
    /**
     * @brief Default construct an MPI communicator wrapper using `MPI_COMM_WORLD`.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does nothing.
     */
    communicator();

#if defined(PLSSVM_HAS_MPI_ENABLED)
    /**
     * @brief Construct an MPI communicator wrapper using the provided MPI communicator.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does nothing.
     * @param[in] comm the provided MPI communicator
     * @note This function does not take ownership of the provided MPI communicator!
     */
    explicit communicator(MPI_Comm comm);
#endif

    /**
     * @brief Return the total number of MPI ranks in this communicator.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `0`.
     * @return the number of MPI ranks in this communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t size() const;
    /**
     * @brief Return the current MPI rank with respect to this communicator.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `0`.
     * @return the current MPI rank with respect to this communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t rank() const;
    /**
     * @brief Returns `true` if the current MPI rank is rank `0`, i.e., the main MPI rank.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `true`.
     * @return `true` if the current MPI rank is `0`, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_main_rank() const;
    /**
     * @brief Waits for all MPI ranks in this communicator to finish.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does nothing.
     */
    void barrier() const;

#if defined(PLSSVM_HAS_MPI_ENABLED)
    /**
     * @brief Add implicit conversion operator back to a native MPI communicator.
     * @return The wrapped MPI communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] operator MPI_Comm() const { return comm_; }
#endif

  private:
#if defined(PLSSVM_HAS_MPI_ENABLED)
    /// The wrapped MPI communicator. Only available if `PLSSVM_HAS_MPI_ENABLED` is defined!
    MPI_Comm comm_{ MPI_COMM_WORLD };
#endif
};

}  // namespace plssvm::mpi

#endif  // PLSSVM_MPI_COMMUNICATOR_HPP_

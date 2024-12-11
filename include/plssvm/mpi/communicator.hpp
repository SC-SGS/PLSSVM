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

#include "plssvm/mpi/detail/mpi_datatype.hpp"  // plssvm::mpi::detail::mpi_datatype
#include "plssvm/mpi/detail/utility.hpp"       // PLSSVM_MPI_ERROR_CHECK

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_Comm, MPI_COMM_WORLD, MPI_Gather
#endif

#include <algorithm>    // std::transform
#include <cstddef>      // std::size_t
#include <functional>   // std::invoke
#include <type_traits>  // std::is_enum_v, std::underlying_type_t
#include <vector>       // std::vector

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
     * @brief Return the MPI rank that is identified as main MPI rank.
     * @details For PLSSVM, the main MPI rank is rank `0` in the current communicator.
     * @return the main MPI rank `0` (`[[nodiscard]]`)
     */
    [[nodiscard]] static std::size_t main_rank() { return 0; }

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

    /**
     * @brief Execute the provided function @p f in a sequential manner across all MPI ranks in the current MPI communicator.
     * @tparam Func the type of the function
     * @param[in] f the function to execute
     */
    template <typename Func>
    void sequentialize(Func f) const {
        // iterate over all potential MPI ranks in the current communicator
        for (std::size_t rank = 0; rank < this->size(); ++rank) {
            // call function only if MY rank matches the current iteration
            if (rank == this->rank()) {
                std::invoke(f);
            }
            // wait for the current MPI rank to finish
            this->barrier();
        }
    }

    /**
     * @brief Gather the @p value from each MPI rank on the `communicator::main_rank()`.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns the provided @p value wrapped in a `std::vector`.
     * @tparam T the type of the values to gather
     * @param value the value to gather at the main MPI rank
     * @return a `std::vector` containing all gathered values (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] std::vector<T> gather(T value) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        std::vector<T> result(this->size());
        PLSSVM_MPI_ERROR_CHECK(MPI_Gather(&value, 1, detail::mpi_datatype<T>(), result.data(), 1, detail::mpi_datatype<T>(), communicator::main_rank(), comm_));
        return result;
#else
        return { value };
#endif
    }

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

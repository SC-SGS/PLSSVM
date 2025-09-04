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

#include "plssvm/detail/utility.hpp"           // PLSSVM_IS_DEFINED
#include "plssvm/matrix.hpp"                   // plssvm::matrix, plssvm::layout_type
#include "plssvm/mpi/detail/mpi_datatype.hpp"  // plssvm::mpi::detail::mpi_datatype
#include "plssvm/mpi/detail/utility.hpp"       // PLSSVM_MPI_ERROR_CHECK

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_Comm, MPI_COMM_WORLD, MPI_Gather, MPI_Allreduce, MPI_Exscan, MPI_IN_PLACE, MPI_SUM
#endif

#include <chrono>      // std::chrono::milliseconds
#include <cstddef>     // std::size_t
#include <functional>  // std::invoke
#include <optional>    // std::optional
#include <string>      // std::string
#include <vector>      // std::vector

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
    communicator() = default;

    /**
     * @brief Default construct an MPI communicator wrapper using `MPI_COMM_WORLD` and set the load balancing @p weights.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, only stores the load balancing weights.
     * @param[in] weights the load balancing weights
     * @throws plssvm::mpi_exception if the number of @p weights does not match the MPI communicator size
     */
    explicit communicator(std::vector<std::size_t> weights);

#if defined(PLSSVM_HAS_MPI_ENABLED)
    /**
     * @brief Construct an MPI communicator wrapper using the provided MPI communicator.
     * @param[in] comm the provided MPI communicator
     * @note This function does not take ownership of the provided MPI communicator!
     */
    explicit communicator(MPI_Comm comm);

    /**
     * @brief Construct an MPI communicator wrapper using the provided MPI communicator and set the load balancing @p weights.
     * @param[in] comm the provided MPI communicator
     * @param[in] weights the load balancing weights
     * @throws plssvm::mpi_exception if the number of @p weights does not match the MPI communicator size
     * @note This function does not take ownership of the provided MPI communicator!
     */
    communicator(MPI_Comm comm, std::vector<std::size_t> weights);
#endif

    /**
     * @brief Return the total number of MPI ranks in this communicator.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns `1`.
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
    [[nodiscard]] constexpr static std::size_t main_rank() { return 0; }

    /**
     * @brief Check whether distributed execution via MPI is enabled.
     * @return `true` if MPI is enabled, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static bool is_mpi_enabled() { return PLSSVM_IS_DEFINED(PLSSVM_HAS_MPI_ENABLED); }

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
     * @details The order is determined by the MPI ranks' values.
     * @tparam Func the type of the function
     * @param[in] f the function to execute
     */
    template <typename Func>
    void serialize(Func f) const {
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
     * @param[in] value the value to gather at the main MPI rank
     * @return a `std::vector` containing all gathered values (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] std::vector<T> gather(T value) const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        std::vector<T> result(this->size());
        PLSSVM_MPI_ERROR_CHECK(MPI_Gather(&value, 1, detail::mpi_datatype<T>(), result.data(), 1, detail::mpi_datatype<T>(), communicator::main_rank(), comm_));
        return result;
#else
        return { value };
#endif
    }

    /**
     * @brief Gather the `std::string` @p str from each MPI rank on the `communicator::main_rank()`.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns the provided @p str wrapped in a `std::vector`.
     * @param[in] str the string to gather at the main MPI rank
     * @return a `std::vector` containing all gathered strings (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::string> gather(const std::string &str) const;

    /**
     * @brief Gather the `std::chrono::milliseconds` @p duration from each MPI rank on the `communicator::main_rank()`.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns the provided @p duration wrapped in a `std::vector`.
     * @param[in] duration the duration to gather at the main MPI rank
     * @return a `std::vector` containing all gathered durations (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::chrono::milliseconds> gather(const std::chrono::milliseconds &duration) const;

    /**
     * @brief Gather the @p value from each MPI rank and distribute the result to all MPI ranks.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, returns the provided @p value wrapped in a `std::vector`.
     * @tparam T the type of the values to gather
     * @param[in] value the value to gather on all MPI ranks
     * @return a `std::vector` containing all gathered values (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] std::vector<T> allgather(T value) const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        std::vector<T> result(this->size());
        PLSSVM_MPI_ERROR_CHECK(MPI_Allgather(&value, 1, detail::mpi_datatype<T>(), result.data(), 1, detail::mpi_datatype<T>(), comm_));
        return result;
#else
        return { value };
#endif
    }

    /**
     * @brief Reduce the @p matr on all MPI ranks by summing all elements elementwise.
     * @details If `PLSSVM_HAS_MPI_ENABLED` is undefined, does not mutate `matr`.
     * @tparam T the value type of the matrix
     * @tparam layout the matrix layout
     * @param[in,out] matr the matrix to reduce, changed inplace
     */
    template <typename T, layout_type layout>
    void allreduce_inplace([[maybe_unused]] plssvm::matrix<T, layout> &matr) const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        PLSSVM_MPI_ERROR_CHECK(MPI_Allreduce(MPI_IN_PLACE, matr.data(), static_cast<int>(matr.size_padded()), detail::mpi_datatype<T>(), MPI_SUM, comm_));
#endif
    }

#if defined(PLSSVM_HAS_MPI_ENABLED)
    /**
     * @brief Add implicit conversion operator back to a native MPI communicator.
     * @return The wrapped MPI communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] operator MPI_Comm() const { return comm_; }
#endif

    /**
     * @brief Update the load balancing weights.
     * @param[in] weights the new weights
     * @throws plssvm::mpi_exception if the number of @p weights does not match the MPI communicator size
     */
    void set_load_balancing_weights(std::vector<std::size_t> weights);

    /**
     * @brief Return the current load balancing weights if any.
     * @details If assertions are enabled and there are load balancing weights, always checks whether the load balancing weights are the same for **all** MPI ranks.
     * @return the weights (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::optional<std::vector<std::size_t>> &get_load_balancing_weights() const noexcept;

    /**
     * @brief Check whether @p lhs and @p rhs are equal, i.e., they are identical to each other, otherwise no collective operations are supported.
     * @param[in] lhs the first MPI communicator
     * @param[in] rhs the second MPI communicator
     * @return `true` if both communicators are identical, otherwise `false`
     */
    friend bool operator==(const communicator &lhs, const communicator &rhs) noexcept;
    /**
     * @brief Check whether @p lhs and @p rhs are unequal, i.e., they are **not** identical to each other.
     * @param[in] lhs the first MPI communicator
     * @param[in] rhs the second MPI communicator
     * @return `true` if both communicators are **not** identical, otherwise `false`
     */
    friend bool operator!=(const communicator &lhs, const communicator &rhs) noexcept;

  private:
#if defined(PLSSVM_HAS_MPI_ENABLED)
    /// The wrapped MPI communicator. Only available if `PLSSVM_HAS_MPI_ENABLED` is defined!
    MPI_Comm comm_{ MPI_COMM_WORLD };
#endif
    /// The MPI load balancing weights. Always guaranteed to be the same size as the communicator size.
    std::optional<std::vector<std::size_t>> load_balancing_weights_{ std::nullopt };
};

}  // namespace plssvm::mpi

#endif  // PLSSVM_MPI_COMMUNICATOR_HPP_

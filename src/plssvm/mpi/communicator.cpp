/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/mpi/communicator.hpp"

#include "plssvm/detail/assert.hpp"            // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"    // plssvm::mpi_exception
#include "plssvm/mpi/detail/mpi_datatype.hpp"  // plssvm::mpi::detail::mpi_datatype
#include "plssvm/mpi/detail/utility.hpp"       // PLSSVM_MPI_ERROR_CHECK

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_Comm, MPI_Comm_size, MPI_Comm_rank, MPI_Barrier, MPI_Gatherv, MPI_Gather, MPI_Bcast
#endif

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::transform
#include <chrono>     // std::chrono::milliseconds
#include <cstddef>    // std::size_t
#include <cstdint>    // std::int64_t
#include <optional>   // std::optional, std::nullopt
#include <string>     // std::string
#include <utility>    // std::move
#include <vector>     // std::vector

namespace plssvm::mpi {

communicator::communicator() :
    load_balancing_weights_{ std::nullopt } { }

communicator::communicator(std::vector<std::size_t> weights) {
    // set load balancing weights
    this->set_load_balancing_weights(std::move(weights));
}

#if defined(PLSSVM_HAS_MPI_ENABLED)
communicator::communicator(MPI_Comm comm) :
    comm_{ comm },
    load_balancing_weights_{ std::nullopt } { }

communicator::communicator(MPI_Comm comm, std::vector<std::size_t> weights) :
    comm_{ comm } {
    // set load balancing weights
    this->set_load_balancing_weights(std::move(weights));
}
#endif

std::size_t communicator::size() const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    int size{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Comm_size(comm_, &size));
    return static_cast<std::size_t>(size);
#else
    return std::size_t{ 0 };
#endif
}

std::size_t communicator::rank() const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    int rank{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Comm_rank(comm_, &rank));
    return static_cast<std::size_t>(rank);
#else
    return std::size_t{ 0 };
#endif
}

bool communicator::is_main_rank() const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    return this->rank() == communicator::main_rank();
#else
    return true;
#endif
}

void communicator::barrier() const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    PLSSVM_MPI_ERROR_CHECK(MPI_Barrier(comm_));
#endif
}

std::vector<std::string> communicator::gather(const std::string &str) const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    // gather string size information
    const std::vector<int> sizes = this->gather(static_cast<int>(str.size()));

    // calculate displacements and create receive-buffer (on main rank only!)
    std::vector<char> recv_buffer{};
    std::vector<int> displacements(sizes.size());
    if (this->is_main_rank()) {
        int total_size{};
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            displacements[i] = total_size;
            total_size += sizes[i];
        }
        recv_buffer.resize(total_size);
    }

    // gather the strings on the MPI main rank
    PLSSVM_MPI_ERROR_CHECK(MPI_Gatherv(str.data(), str.size(), detail::mpi_datatype<char>(), recv_buffer.data(), sizes.data(), displacements.data(), detail::mpi_datatype<char>(), communicator::main_rank(), comm_));

    // unpack the receive-buffer to the separate strings
    std::vector<std::string> result(sizes.size());
    if (this->is_main_rank()) {
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            result[i] = std::string(recv_buffer.begin() + displacements[i],
                                    recv_buffer.begin() + displacements[i] + sizes[i]);
        }
    }
    return result;
#else
    return { str };
#endif
}

std::vector<std::chrono::milliseconds> communicator::gather(const std::chrono::milliseconds &duration) const {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    // convert the duration to an integer
    const std::int64_t intermediate_dur = duration.count();
    std::vector<std::int64_t> intermediate_result(this->size());
    // gather the integer values from each MPI rank
    PLSSVM_MPI_ERROR_CHECK(MPI_Gather(&intermediate_dur, 1, detail::mpi_datatype<std::int64_t>(), intermediate_result.data(), 1, detail::mpi_datatype<std::int64_t>(), communicator::main_rank(), comm_));
    // cast integers back to durations
    std::vector<std::chrono::milliseconds> result(this->size());
    std::transform(intermediate_result.cbegin(), intermediate_result.cend(), result.begin(), [](const std::int64_t dur) { return static_cast<std::chrono::milliseconds>(dur); });
    return result;
#else
    return { duration };
#endif
}

void communicator::set_load_balancing_weights(std::vector<std::size_t> weights) {
    if (weights.size() != this->size()) {
        throw mpi_exception{ fmt::format("The number of load balancing weights ({}) must match the number of MPI ranks ({})!", weights.size(), this->size()) };
    }
    load_balancing_weights_ = std::move(weights);
}

const std::optional<std::vector<std::size_t>> &communicator::get_load_balancing_weights() const noexcept {
#if defined(PLSSVM_ENABLE_ASSERTS) && defined(PLSSVM_HAS_MPI_ENABLED)
    // check if all MPI ranks have balancing weights
    bool has_weights = load_balancing_weights_.has_value();
    bool and_result{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Allreduce(&has_weights, &and_result, 1, MPI_C_BOOL, MPI_LAND, comm_));
    bool or_result{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Allreduce(&has_weights, &or_result, 1, MPI_C_BOOL, MPI_LOR, comm_));

    // All ranks are true: Both MPI_LAND and MPI_LOR will return 1.
    // All ranks are false: Both MPI_LAND and MPI_LOR will return 0.
    // Mixed values: MPI_LAND will return 0, and MPI_LOR will return 1.
    // -> if the values are not equal some ranks have load balancing weights and some don't
    PLSSVM_ASSERT(and_result == or_result, "Some MPI ranks have load balancing weights and some don't!");

    // if all MPI ranks have load balancing weights, check that they are the same
    if (and_result) {
        // check that the balancing weights are the same for all MPI ranks
        std::vector<std::size_t> reference_weights(load_balancing_weights_->size());
        if (this->is_main_rank()) {
            reference_weights = load_balancing_weights_.value();
        }
        PLSSVM_MPI_ERROR_CHECK(MPI_Bcast(reference_weights.data(), reference_weights.size(), detail::mpi_datatype<std::size_t>(), communicator::main_rank(), comm_));
        // each rank checks whether its array is correct
        // if this is not the case for at least one array, abort
        PLSSVM_ASSERT(static_cast<bool>(reference_weights == load_balancing_weights_.value()), "The load balancing weights must be the same on all MPI ranks which is currently not the case!");
    }
#endif
    return load_balancing_weights_;
}

}  // namespace plssvm::mpi

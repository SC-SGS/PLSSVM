/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/mpi/communicator.hpp"

#include "plssvm/mpi/detail/utility.hpp"  // PLSSVM_MPI_ERROR_CHECK

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"
#endif

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::mpi {

communicator::communicator() { }

#if defined(PLSSVM_HAS_MPI_ENABLED)
communicator::communicator(MPI_Comm comm) :
    comm_{ comm } { }
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
    PLSSVM_MPI_ERROR_CHECK(MPI_Gatherv(str.data(), str.size(), MPI_CHAR, recv_buffer.data(), sizes.data(), displacements.data(), MPI_CHAR, communicator::main_rank(), comm_));

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

}  // namespace plssvm::mpi

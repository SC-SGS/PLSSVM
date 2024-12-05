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
    return this->rank() == std::size_t{ 0 };
#else
    return false;
#endif
}

}  // namespace plssvm::mpi
